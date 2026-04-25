A. 语音指令测试集

准备 30～50 条语音命令，分成：

普通动作指令：20 条
组合动作指令：10 条
急停/停止指令：10 条
无效/模糊指令：10 条

| 论文指标                            | 你测试脚本应该测什么                        |
| ------------------------------- | --------------------------------- |
| ASR transcription success rate  | Whisper 返回文本是否可用                  |
| command-level semantic accuracy | ASR 文本是否保留原命令意图                   |
| emergency routing accuracy      | emergency 样本是否被 route 到 emergency |
| plan validity rate              | LLM 输出 JSON / ActionTask 是否格式合法   |
| intent-plan matching accuracy   | LLM 输出动作是否符合 expected_steps       |
| planning latency                | transcript → plan 完成              |
| TTFT                            | planner request → first token     |
| voice-to-plan latency           | utterance end → plan 完成           |
| voice-to-queue latency          | utterance end → TaskQueue 插入      |











ASR transcription success rate
Command-level semantic accuracy
Emergency routing accuracy
Plan validity rate
Intent-plan matching accuracy
Voice-to-plan latency
Voice-to-queue latency
B. 手势反射测试集

准备每类手势若干段视频或实时动作。

测：

Gesture classification accuracy
Precision / Recall / F1-score
Event trigger precision / recall
False trigger rate
Perception-to-queue latency
C. 执行安全测试集

准备非法动作组合 20～30 条。

例如：

PRONE → WALK
LOCOMOTION → ROLL_OVER
IDLE → HIGH_SPEED_MOVE
STAND_DOWN → FORWARD
NORMAL TASK QUEUE → EMERGENCY STOP

测：

FSM correct rejection rate
Queue flush success rate
Emergency preemption latency
Task completion rate
D. vLLM 对比测试集

准备 20 条文本指令，不经过 ASR，直接测试 planner。

测：

TTFT
Full planning latency
Plan validity rate
Intent-plan matching accuracy

对比：

vLLM
naive local generation
五、最终建议你论文里保留的指标表

你可以在论文里最终汇总成这个表：

\begin{table}[htbp]
\centering
\begin{tabular}{|l|l|l|}
\hline
\textbf{Subsystem} & \textbf{Metric} & \textbf{Purpose} \\
\hline
Voice-to-Plan & ASR transcription success rate & Evaluate speech-to-text usability \\
Voice-to-Plan & Command-level semantic accuracy & Evaluate whether command intent is preserved \\
Voice-to-Plan & Emergency routing accuracy & Evaluate safety-critical command bypass \\
Voice-to-Plan & Plan validity rate & Evaluate schema-constrained LLM output \\
Voice-to-Plan & Intent-plan matching accuracy & Evaluate semantic correctness of generated plans \\
Voice-to-Plan & Voice-to-plan latency & Evaluate cognitive response time \\
\hline
Reflex Perception & Gesture classification accuracy & Evaluate gesture recognition correctness \\
Reflex Perception & F1-score & Evaluate balance between missed and false triggers \\
Reflex Perception & Event trigger precision/recall & Evaluate reliability after confidence thresholding \\
Reflex Perception & Perception-to-queue latency & Evaluate reflex pathway responsiveness \\
\hline
Execution Guardian & Task completion rate & Evaluate execution reliability \\
Execution Guardian & FSM correct rejection rate & Evaluate unsafe transition filtering \\
Execution Guardian & Emergency preemption latency & Evaluate interrupt response \\
Execution Guardian & End-to-end response latency & Evaluate complete closed-loop delay \\
\hline
LLM Serving & TTFT & Evaluate first-token responsiveness \\
LLM Serving & Full planning latency & Evaluate complete generation time \\
\hline
\end{tabular}
\caption{\textbf{Evaluation metrics for the proposed embodied control framework.}}
\label{tab:evaluation_metrics}
\end{table}
六、优先级总结

你现在先测这些：

P0 必须：
1. Plan validity rate
2. Intent-plan matching accuracy
3. Emergency routing accuracy
4. Voice-to-plan latency
5. Gesture F1-score
6. Event trigger precision/recall
7. FSM correct rejection rate
8. Emergency preemption latency

P1 强烈建议：
9. TTFT
10. Full planning latency
11. Voice-to-queue latency
12. Perception-to-queue latency
13. Task completion rate

P2 可选：
14. VAD false trigger rate
15. ASR WER/CER
16. GPU memory usage
17. Long-running stability

这样测出来，你的 Results and Discussion 就会非常完整。



