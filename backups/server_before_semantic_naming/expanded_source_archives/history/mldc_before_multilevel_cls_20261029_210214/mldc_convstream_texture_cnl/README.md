# MLDC+CNL：唯一融合学生表示

指定副本仍为 `mldc_convstream_texture_cnl`。当前版本 `mldc_pad_fused_student_20261029`。学生身份表示统一为最终768维融合描述子；不复用ViT注意力投影，不读取ViT原生注意力作为教师。

## 当前计算路径

1. 原Stem＋ViT返回CLS1–12与最后层P12（128×768）。
2. 原MLDC融合CLS9–12得到g。现有3层MHCA＋FFN读取P12，产生内部上下文c，形成fS=g＋Wc。W保持零初始化。
3. 唯一学生描述子zS=normalize(BN768(fS))。原ID、Triplet、CNL和学生原型CE均使用这个身份表示的相应原有训练位置。
4. 学生注意力由同一个zS重新作为查询生成，不再蒸馏融合前上下文的注意力。
5. 教师以768维身份原型为查询，读取同一P12，生成注意力AT及内部上下文cT。教师特征fT=g＋WcT，再经过同一个BN768和L2，得到同一身份空间的zT。
6. 按图像、层、头计算KL(AT.detach() || AS)后平均。AMU在教师启用后由zT更新；构建阶段使用zS。

256只表示注意力计算宽度，不再有独立256维FC＋BN身份嵌入、256维学生身份CE或256维原型表。原型表为500×768。三个深度位置参数各自独立，在融合上下文、融合学生注意力、原型教师路径间共享；训练有教师时每层调用三次。

学生和教师的最后BN共享；学生一次、有效教师一次。标准推理只构建融合特征并使用已保存的BN统计；无需运行学生注意力读出，也不读取身份原型。请求 `return_pad_details=True` 时提供学生注意力等诊断。

## 配置

默认 `configs/pad_p12_nightreid.yml`：
- `PAD.STUDENT_FEATURE: fused`
- `PAD.PROTO_DIM: 768`
- `PAD.DIM: 256`（注意力内部宽度）
- AMU、gamma=0.05、注意力权重5、无递增，保留5轮原型构建。
- NightReID、seed1234、120轮、原20轮学习率预热、原SSL初始化、SIE关闭、CNL权重0.1及阈值0.9。

四个对照配置保留。新输出目录带 `fused_student` 标识，旧结果不覆盖。旧256维原型版本检查点不兼容本次结构；使用原SSL初始化新训练。

## 运行

从项目根目录执行：

~~~text
python -m mldc_convstream_texture_cnl.verify
python -m mldc_convstream_texture_cnl.train --config_file mldc_convstream_texture_cnl/configs/pad_p12_nightreid.yml
python -m mldc_convstream_texture_cnl.test --config_file mldc_convstream_texture_cnl/configs/pad_p12_nightreid.yml TEST.WEIGHT /path/to/new_checkpoint.pth
~~~

修改本身不会启动训练。验证无优化器训练步骤，覆盖真实跨相机图片、唯一描述子、KL对最终融合节点的梯度、原型/记忆提交、导出与AMU公式。详见[方法说明](docs/PAD_METHOD.md)。
