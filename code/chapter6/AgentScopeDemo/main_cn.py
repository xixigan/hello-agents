# -*- coding: utf-8 -*-
"""
三国狼人杀 - 基于AgentScope的中文版狼人杀游戏
融合三国演义角色和传统狼人杀玩法
"""
import asyncio
import os
import random
from typing import List, Dict, Optional

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.pipeline import MsgHub, sequential_pipeline, fanout_pipeline
from agentscope.formatter import OpenAIChatFormatter

# 自定义模型类，添加速率限制重试机制
class RetryOpenAIChatModel(OpenAIChatModel):
    async def __call__(self, messages, tools=None, tool_choice=None, structured_model=None, **kwargs):
        import asyncio
        from openai import RateLimitError as OpenAIRateLimitError
        import httpx
        import httpcore
        
        max_retries = 5
        retry_delay = 1  # 初始延迟1秒
        fixed_delay = 0.5  # 每次请求后添加的固定延迟
        
        for attempt in range(max_retries):
            try:
                # 基于当前环境检测后端类型，决定是否需要严格清理不兼容字段（如 prefix / required）
                llm_model_id = os.environ.get("LLM_MODEL_ID", "").lower()
                llm_base_url = os.environ.get("LLM_BASE_URL", "").lower()

                # 如果不是官方 OpenAI，或显式是第三方（例如 THUDM / GLM / siliconflow），启用严格清理模式
                is_openai_provider = ("openai" in llm_base_url) or llm_model_id.startswith("gpt") or "openai" in llm_model_id
                strict_clean = ("siliconflow" in llm_base_url) or ("thudm" in llm_model_id) or ("glm" in llm_model_id)

                prefix_keys = {"prefix", "prefix_text", "instruction_prefix", "explain_prefix"}

                def _recursive_clean(obj, remove_required: bool = False) -> bool:
                    """递归清理 dict/list 中的前缀字段和（可选）required 字段，返回是否有移除操作"""
                    removed_any = False
                    if isinstance(obj, dict):
                        for k in list(obj.keys()):
                            if k in prefix_keys:
                                obj.pop(k, None)
                                removed_any = True
                            elif remove_required and k == "required":
                                obj.pop(k, None)
                                removed_any = True
                            else:
                                try:
                                    sub = obj.get(k)
                                    if _recursive_clean(sub, remove_required):
                                        removed_any = True
                                except Exception:
                                    pass
                    elif isinstance(obj, list):
                        for item in obj:
                            if _recursive_clean(item, remove_required):
                                removed_any = True
                    return removed_any

                # 在严格模式下，递归清理 tool_choice、tools、kwargs 中的前缀和 required 字段
                if strict_clean and not is_openai_provider:
                    removed = False
                    if isinstance(tool_choice, dict):
                        if _recursive_clean(tool_choice, remove_required=True):
                            removed = True
                    if isinstance(tools, list):
                        for t in tools:
                            if _recursive_clean(t, remove_required=True):
                                removed = True
                    # 对 kwargs 全面清理，防止当 structured_model/formatter 生成了不兼容选项
                    if _recursive_clean(kwargs, remove_required=True):
                        removed = True
                    if removed:
                        print("⚠️ 检测到并移除了与第三方模型不兼容的 'prefix' 或 'required' 字段")
                else:
                    # 非严格模式下，也仅清理 json_schema 下的 prefix 字段，避免普通字段被误删
                    rf = kwargs.get("response_format")
                    removed = False
                    if isinstance(rf, dict) and rf.get("type") == "json_schema":
                        if _recursive_clean(rf, remove_required=False):
                            removed = True
                        kwargs["response_format"] = rf
                    for k, v in list(kwargs.items()):
                        if isinstance(v, dict) and v.get("type") == "json_schema":
                            if _recursive_clean(v, remove_required=False):
                                removed = True
                            kwargs[k] = v
                    if removed:
                        print("⚠️ 检测到并移除了 response_format 中的前缀字段，以兼容 json_schema 模式")

                # 在调用父类前，如果提供了 structured_model（如 Pydantic 模型），则显式构造不含 'prefix' 的 response_format 以防第三方后端拒绝带 prefix 的 json_schema
                try:
                    if structured_model is not None:
                        schema = None
                        # 如果 structured_model 是 Pydantic 模型类或实例，使用 .schema() 获取 json-schema
                        if hasattr(structured_model, "schema") and callable(getattr(structured_model, "schema")):
                            schema = structured_model.schema()
                        # 如果 structured_model 是已生成的 dict schema 或其它 dict，也支持直接使用
                        if isinstance(schema, dict):
                            # 覆盖或设置 response_format，确保没有 'prefix' 字段
                            kwargs["response_format"] = {"type": "json_schema", "json_schema": schema}
                except Exception:
                    pass

                # 调用父类的__call__方法，并在遇到 BadRequestError（如后端拒绝 json_schema 中的 prefix）时尝试回退并重试一次
                from openai import BadRequestError
                try:
                    result = await super().__call__(messages, tools, tool_choice, structured_model, **kwargs)
                except BadRequestError as e:
                    msg = str(e)
                    # 针对 20015 / prefix 不被允许 的错误进行特定回退
                    if ("prefix" in msg and "json_schema" in msg) or "20015" in msg:
                        print("⚠️ 后端返回 prefix 不允许错误，正在移除所有 'prefix' 字段并重试一次...")
                        try:
                            if isinstance(tool_choice, dict):
                                _recursive_clean(tool_choice, remove_required=True)
                            if isinstance(tools, list):
                                for t in tools:
                                    _recursive_clean(t, remove_required=True)
                            _recursive_clean(kwargs, remove_required=True)
                            # 如果有 structured_model，确保 response_format 使用干净的 schema
                            if structured_model is not None and hasattr(structured_model, "schema") and callable(getattr(structured_model, "schema")):
                                schema = structured_model.schema()
                                if isinstance(schema, dict):
                                    kwargs["response_format"] = {"type": "json_schema", "json_schema": schema}
                        except Exception:
                            pass
                        # 重试一次（如再次失败则抛出）
                        result = await super().__call__(messages, tools, tool_choice, structured_model, **kwargs)
                    else:
                        raise
                
                # 在每次成功请求后添加固定延迟，减少API调用频率
                await asyncio.sleep(fixed_delay)
                return result
            except (OpenAIRateLimitError, httpx.TimeoutException, httpcore.TimeoutException, httpx.ConnectError):
                if attempt == max_retries - 1:
                    # 最后一次重试失败，重新抛出异常
                    raise
                
                # 指数退避重试
                print(f"⚠️  遇到速率限制或连接问题，{retry_delay}秒后重试... (第{attempt + 1}/{max_retries}次)")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # 指数增加延迟时间
                retry_delay = min(retry_delay, 60)  # 最大延迟不超过60秒

from prompt_cn import ChinesePrompts
from game_roles import GameRoles
from structured_output_cn import (
    DiscussionModelCN,
    get_vote_model_cn,
    WitchActionModelCN,
    get_seer_model_cn,
    get_hunter_model_cn,
    WerewolfKillModelCN
)
from utils_cn import (
    check_winning_cn,
    majority_vote_cn,
    get_chinese_name,
    format_player_list,
    GameModerator,
    MAX_GAME_ROUND,
    MAX_DISCUSSION_ROUND,
)


class ThreeKingdomsWerewolfGame:
    """三国狼人杀游戏主类"""
    
    def __init__(self):
        self.players: Dict[str, ReActAgent] = {}
        self.roles: Dict[str, str] = {}
        self.moderator = GameModerator()
        self.alive_players: List[ReActAgent] = []
        self.werewolves: List[ReActAgent] = []
        self.villagers: List[ReActAgent] = []
        self.seer: List[ReActAgent] = []
        self.witch: List[ReActAgent] = []
        self.hunter: List[ReActAgent] = []
        
        # 女巫道具状态
        self.witch_has_antidote = True
        self.witch_has_poison = True
        
    async def create_player(self, role: str, character: str) -> ReActAgent:
        """创建具有三国背景的玩家"""
        name = get_chinese_name(character)
        self.roles[name] = role
        
        agent = ReActAgent(
            name=name,
            sys_prompt=ChinesePrompts.get_role_prompt(role, character),
            model=RetryOpenAIChatModel(
                model_name=os.environ["LLM_MODEL_ID"],
                api_key=os.environ["LLM_API_KEY"],
                client_kwargs={
                    "base_url": os.environ["LLM_BASE_URL"],
                    "timeout": 60.0  # 设置60秒超时
                },
            ),
            formatter=OpenAIChatFormatter(),
        )
        
        # 角色身份确认
        await agent.observe(
            await self.moderator.announce(
                f"【{name}】你在这场三国狼人杀中扮演{GameRoles.get_role_desc(role)}，"
                f"你的角色是{character}。{GameRoles.get_role_ability(role)}"
            )
        )
        
        self.players[name] = agent
        return agent
    
    async def setup_game(self, player_count: int = 6):
        """设置游戏"""
        print("🎮 开始设置三国狼人杀游戏...")
        
        # 获取角色配置
        roles = GameRoles.get_standard_setup(player_count)
        characters = random.sample([
            "刘备", "关羽", "张飞", "诸葛亮", "赵云",
            "曹操", "司马懿", "周瑜", "孙权"
        ], player_count)
        
        # 创建玩家
        for i, (role, character) in enumerate(zip(roles, characters)):
            agent = await self.create_player(role, character)
            self.alive_players.append(agent)
            
            # 分配到对应阵营
            if role == "狼人":
                self.werewolves.append(agent)
            elif role == "预言家":
                self.seer.append(agent)
            elif role == "女巫":
                self.witch.append(agent)
            elif role == "猎人":
                self.hunter.append(agent)
            else:
                self.villagers.append(agent)
        
        # 游戏开始公告
        await self.moderator.announce(
            f"三国狼人杀游戏开始！参与者：{format_player_list(self.alive_players)}"
        )
        
        print(f"✅ 游戏设置完成，共{len(self.alive_players)}名玩家")
    
    async def werewolf_phase(self, round_num: int):
        """狼人阶段"""
        if not self.werewolves:
            return None
            
        await self.moderator.announce(f"🐺 狼人请睁眼，选择今晚要击杀的目标...")
        
        # 狼人讨论
        async with MsgHub(
            self.werewolves,
            enable_auto_broadcast=True,
            announcement=await self.moderator.announce(
                f"狼人们，请讨论今晚的击杀目标。存活玩家：{format_player_list(self.alive_players)}"
            ),
        ) as werewolves_hub:
            # 讨论阶段
            for _ in range(MAX_DISCUSSION_ROUND):
                for wolf in self.werewolves:
                    await wolf(structured_model=DiscussionModelCN)
            
            # 投票击杀
            werewolves_hub.set_auto_broadcast(False)
            kill_votes = await fanout_pipeline(
                self.werewolves,
                msg=await self.moderator.announce("请选择击杀目标"),
                structured_model=WerewolfKillModelCN,
                enable_gather=False,
            )
            
            # 统计投票
            votes = {}
            for i, vote_msg in enumerate(kill_votes):
                # 检查vote_msg是否为None或metadata是否存在
                if vote_msg is not None and hasattr(vote_msg, 'metadata') and vote_msg.metadata is not None:
                    votes[self.werewolves[i].name] = vote_msg.metadata.get("target")
                else:
                    # 如果返回无效,随机选择一个目标
                    print(f"⚠️ {self.werewolves[i].name} 的击杀投票无效,随机选择目标")
                    import random
                    valid_targets = [p.name for p in self.alive_players if p.name not in [w.name for w in self.werewolves]]
                    votes[self.werewolves[i].name] = random.choice(valid_targets) if valid_targets else None
            
            killed_player, _ = majority_vote_cn(votes)
            return killed_player
    
    async def seer_phase(self):
        """预言家阶段"""
        if not self.seer:
            return
            
        seer_agent = self.seer[0]
        await self.moderator.announce("🔮 预言家请睁眼，选择要查验的玩家...")
        
        check_result = await seer_agent(
            structured_model=get_seer_model_cn(self.alive_players)
        )

        # 检查返回结果是否有效
        if check_result is None or not hasattr(check_result, 'metadata') or check_result.metadata is None:
            print(f"⚠️ 预言家查验失败,跳过此阶段")
            return

        target_name = check_result.metadata.get("target")
        if not target_name:
            print(f"⚠️ 预言家未选择查验目标,跳过此阶段")
            return

        target_role = self.roles.get(target_name, "村民")
        
        # 告知预言家结果
        result_msg = f"查验结果：{target_name}是{'狼人' if target_role == '狼人' else '好人'}"
        await seer_agent.observe(await self.moderator.announce(result_msg))
    
    async def witch_phase(self, killed_player: str):
        """女巫阶段"""
        if not self.witch:
            return killed_player, None
            
        witch_agent = self.witch[0]
        await self.moderator.announce("🧙‍♀️ 女巫请睁眼...")
        
        # 告知女巫死亡信息
        death_info = f"今晚{killed_player}被狼人击杀" if killed_player else "今晚平安无事"
        await witch_agent.observe(await self.moderator.announce(death_info))
        
        # 女巫行动
        witch_action = await witch_agent(structured_model=WitchActionModelCN)

        saved_player = None
        poisoned_player = None

        # 检查返回结果是否有效
        if witch_action is None or not hasattr(witch_action, 'metadata') or witch_action.metadata is None:
            print(f"⚠️ 女巫行动失败,视为不使用技能")
        else:
            if witch_action.metadata.get("use_antidote") and self.witch_has_antidote:
                if killed_player:
                    saved_player = killed_player
                    self.witch_has_antidote = False
                    await witch_agent.observe(await self.moderator.announce(f"你使用解药救了{killed_player}"))

            if witch_action.metadata.get("use_poison") and self.witch_has_poison:
                poisoned_player = witch_action.metadata.get("target_name")
                if poisoned_player:
                    self.witch_has_poison = False
                    await witch_agent.observe(await self.moderator.announce(f"你使用毒药毒杀了{poisoned_player}"))
        
        # 确定最终死亡玩家
        final_killed = killed_player if not saved_player else None
        
        return final_killed, poisoned_player
    
    async def hunter_phase(self, shot_by_hunter: str):
        """猎人阶段"""
        if not self.hunter:
            return None
            
        hunter_agent = self.hunter[0]
        if hunter_agent.name == shot_by_hunter:
            await self.moderator.announce("🏹 猎人发动技能，可以带走一名玩家...")
            
            hunter_action = await hunter_agent(
                structured_model=get_hunter_model_cn(self.alive_players)
            )

            # 检查返回结果是否有效
            if hunter_action is None or not hasattr(hunter_action, 'metadata') or hunter_action.metadata is None:
                print(f"⚠️ 猎人技能使用失败,视为放弃开枪")
                return None

            if hunter_action.metadata.get("shoot"):
                target = hunter_action.metadata.get("target")
                if target:
                    await self.moderator.announce(f"猎人{hunter_agent.name}开枪带走了{target}")
                    return target
                else:
                    print(f"⚠️ 猎人选择开枪但未指定目标,视为放弃")
                    return None
        
        return None
    
    def update_alive_players(self, dead_players: List[str]):
        """更新存活玩家列表"""
        for dead_name in dead_players:
            if dead_name:
                # 从存活列表移除
                self.alive_players = [p for p in self.alive_players if p.name != dead_name]
                # 从各阵营移除
                self.werewolves = [p for p in self.werewolves if p.name != dead_name]
                self.villagers = [p for p in self.villagers if p.name != dead_name]
                self.seer = [p for p in self.seer if p.name != dead_name]
                self.witch = [p for p in self.witch if p.name != dead_name]
                self.hunter = [p for p in self.hunter if p.name != dead_name]
    
    async def day_phase(self, round_num: int):
        """白天阶段"""
        await self.moderator.day_announcement(round_num)
        
        # 讨论阶段
        async with MsgHub(
            self.alive_players,
            enable_auto_broadcast=True,
            announcement=await self.moderator.announce(
                f"现在开始自由讨论。存活玩家：{format_player_list(self.alive_players)}"
            ),
        ) as all_hub:
            # 每人发言一轮
            await sequential_pipeline(self.alive_players)
            
            # 投票阶段
            all_hub.set_auto_broadcast(False)
            vote_msgs = await fanout_pipeline(
                self.alive_players,
                await self.moderator.announce("请投票选择要淘汰的玩家"),
                structured_model=get_vote_model_cn(self.alive_players),
                enable_gather=False,
            )
            
            # 统计投票
            votes = {}
            for i, vote_msg in enumerate(vote_msgs):
                # 检查vote_msg是否为None或metadata是否存在
                if vote_msg is not None and hasattr(vote_msg, 'metadata') and vote_msg.metadata is not None:
                    votes[self.alive_players[i].name] = vote_msg.metadata.get("vote")
                else:
                    # 如果返回无效,默认弃票
                    print(f"⚠️ {self.alive_players[i].name} 的投票无效,视为弃票")
                    votes[self.alive_players[i].name] = None
            
            voted_out, vote_count = majority_vote_cn(votes)
            await self.moderator.vote_result_announcement(voted_out, vote_count)
            
            return voted_out
    
    async def run_game(self):
        """运行游戏主循环"""
        try:
            await self.setup_game()
            
            for round_num in range(1, MAX_GAME_ROUND + 1):
                print(f"\n🌙 === 第{round_num}轮游戏开始 ===")
                
                # 夜晚阶段
                await self.moderator.night_announcement(round_num)
                
                # 狼人击杀
                killed_player = await self.werewolf_phase(round_num)
                
                # 预言家查验
                await self.seer_phase()
                
                # 女巫行动
                final_killed, poisoned_player = await self.witch_phase(killed_player)
                
                # 更新死亡玩家
                night_deaths = [p for p in [final_killed, poisoned_player] if p]
                self.update_alive_players(night_deaths)
                
                # 死亡公告
                await self.moderator.death_announcement(night_deaths)
                
                # 检查胜利条件
                winner = check_winning_cn(self.alive_players, self.roles)
                if winner:
                    await self.moderator.game_over_announcement(winner)
                    return
                
                # 白天阶段
                voted_out = await self.day_phase(round_num)
                
                # 猎人技能
                hunter_shot = await self.hunter_phase(voted_out)
                
                # 更新死亡玩家
                day_deaths = [p for p in [voted_out, hunter_shot] if p]
                self.update_alive_players(day_deaths)
                
                # 检查胜利条件
                winner = check_winning_cn(self.alive_players, self.roles)
                if winner:
                    await self.moderator.game_over_announcement(winner)
                    return
                
                print(f"第{round_num}轮结束，存活玩家：{format_player_list(self.alive_players)}")
        
        except Exception as e:
            print(f"❌ 游戏运行出错：{e}")
            import traceback
            traceback.print_exc()


async def main():
    """主函数"""
    # 检查环境变量
    required_env_vars = ["LLM_MODEL_ID", "LLM_API_KEY", "LLM_BASE_URL"]
    for env_var in required_env_vars:
        if env_var not in os.environ:
            print(f"❌ 请设置环境变量 {env_var}")
            return
    
    print("🎮 欢迎来到三国狼人杀！")
    
    # 创建并运行游戏
    game = ThreeKingdomsWerewolfGame()
    await game.run_game()


if __name__ == "__main__":
    asyncio.run(main())
