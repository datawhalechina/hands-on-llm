# 全量微调

# 引言

问题自动生成(Question Generation)作为一个重要的研究课题已经在很多实际应用场景中有落地，通过机器主动提问可以用来高效构建或者补充知识库，扩大数据集规模。问题生成技术已经应用到诸多实际应用场景中，如在医药领域，可以应用到自动问诊、辅助诊疗等场景。

本章将基于[中医文献问题生成数据集](https://tianchi.aliyun.com/dataset/86895)，通过全量微调LLaMA，训练一个问题自动生成模型。
由于原版LLaMA主要针对英语进行训练（具体详见LLaMA论文），对中文的支持不是特别理想。
因此，为了在中文任务上获得更好的模型性能，整个全量微调的训练流程包括词表扩充、预训练和指令精调三部分。

参考：[Towards Better Instruction Following Language Models for Chinese](https://arxiv.org/abs/2304.07854)


# 数据准备
本项目使用的数据集是[中医文献问题生成数据集](https://tianchi.aliyun.com/dataset/86895)，可以直接进行下载。

## 数据介绍

标注数据源来自中医药领域文本，包括【黄帝内经翻译版】、【名医百科中医篇】、【中成药用药卷】、【慢性病养生保健科普知识】四个主要来源，共标注 13000对（问题、文档、答案），来源于5000篇文档，每篇文档由人工标注产生1～4对(问题, 答案)对。3500篇语料将开放出来用做训练数据。
问题类型包括实体类和描述类两大类（是非类问题包含在描述类中），其中问题均由人工标注产生，答案是段落中的文本中的连续片段。

数据集以Json格式提供，示例如下：

```bash
{
    "id": 98,
    "text": "黄帝道：什麽叫重实？岐伯说：所谓重实，如大热病人，邪气甚热，而脉象又盛满，内外俱实，便叫重实",
    "annotations": [
        {
            "Q": "重实是指什么？",
            "A": "所谓重实，如大热病人，邪气甚热，而脉象又盛满，内外俱实，便叫重实"
        },
        {
            "Q": "重实之人的脉象是什么样？",
            "A": "脉象又盛满"
        }
    ],
    "source": "黄帝内经翻译版"
}
```

其中`id`表示段落id。`text`表示文档。
`annotations`表示包含多个问题-答案对，每个问题-答案对中`Q`表示问题，`A`表示答案。
问题生成任务需要根据文档及答案自动生成相应的问题，即实现“文档+答案→问题”的流程。

## 数据格式转换（todo）

在模型训练之前，需要使用convert_to_conv_data.py脚本将数据转换成统一的格式:

```bash
python scripts/convert_to_conv_data.py --orig_data school_math_0.25M.json --write_data school_math_0.25M_conv.json --dataset_name bellemath
```

转换后的数据如下：
```bash
{"id": "bellemath42", "conversations": [{"from": "human", "value": " 一箱鸡蛋有25个，小明爸爸每天给他9个鸡蛋吃，那么一箱鸡蛋能够供小明吃几天？"}, {"from": "assistant", "value": "\n一箱鸡蛋有25个，小明每天吃9个鸡蛋，用除法计算一下即可：\n25 ÷ 9 = 2余7\n也就是说，小明第1天到第2天共吃2个鸡蛋，第3天吃剩下的7个鸡蛋。\n因此，一箱鸡蛋能够供小明吃2天。"}]}
{"id": "bellemath80", "conversations": [{"from": "human", "value": "小明手中有15颗糖果，他想要分给他的朋友小红和小李，每人分得的糖果数是3颗，那么他还剩下几颗糖果？"}, {"from": "assistant", "value": " \n1. 该题可以用减法来解决。小明手中有15颗糖果，每人分得3颗，所以小红和小李一共分得6颗糖果（3+3=6）。\n2. 然后再用原来的糖果总数减去分给小红和小李的糖果数，即可得到小明手中剩余的糖果数。 \n计算过程如下：\n15 - 6 = 9\n所以，小明还剩下9颗糖果。"}]}
```

## 数据切分（todo）

这里选取前1000条作为验证集，其余数据作为训练集
```bash
head -n 1000 school_math_0.25M_conv.json > belleMath-dev1K.json
tail -n +1001 school_math_0.25M_conv.json > belleMath.json
```

通过以上流程，全量微调的数据就准备好了。

# 词表扩充
原版LLaMA模型的词表大小为32K，其中仅包含很少的中文字符。
在词元化（tokenization）时，词元生成器（tokenizer）需要多个词元（token）才能拼成一个完整的汉字，编解码的效率比较低。

为了实现中文的高效词元化，在微调之前，需要对LLaMA的原始词表进行扩充。
具体分成以下几个步骤：

1. 在1200万行的中文文本上基于byte-pair encoding (BPE)算法，训练一个词元生成器，词表大小为50k。训练过程使用的是sentencepiece库。

2. 将经过训练得到的词表与原始LLaMA的词表合并，得到一个新词表，大小为79,458。

通过在5000行中文文本上测试新的和原始的词元生成器，一行的平均词元数量从733个减少到291个，显著提高了编解码的效率。

# 预训练
词表扩充之后，首先需要调整词嵌入（word embedding）的大小，并对这部分参数进行随机初始化。
然后，通过继续预训练的方式，对这部分新加入的参数进行学习。
具体来说，预训练使用了3.4B中文词，只更新词嵌入的参数，保持其他参数固定。

由于预训练需要较多的资源，这里直接提供了预训练好的模型。

注意Facebook官方发布的LLaMA模型禁止商用，并且官方没有正式开源模型权重。
为了遵循相应的许可，目前暂时无法发布完整的模型权重。
这里发布的是增量权重，可以理解为原版LLaMA模型上的一个“补丁”，两者进行合并即可获得完整版权重。

为了得到预训练模型，首先需要下载原版LLaMA模型以及增量模型，然后进行合并。
请参考本项目给出的合并模型步骤重构模型。

## 原版LLaMA模型下载
原版LLaMA模型可以通过[LLaMA项目](https://github.com/facebookresearch/llama)申请使用或参考这个[PR](https://github.com/facebookresearch/llama/pull/73/files)进行下载。
因版权问题，本项目无法提供下载链接。


## 增量模型下载
增量模型可以通过huggingface直接下载。

### 模型下载
```
git clone https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-7B
```

### 检查md5sum
```
md5sum ./*
228a21b7bf927f7ffd44c16c88256684  ./config.json.fb090219f6fed69687ab8f9c902f7802cff8060b08007ca0e5af177a8f9613d5.enc
f9b33d359f17a437f6c24b4de6f2272e  ./generation_config.json.fd7ff399e5568cc21a0a8414f43df88ef7c424995b9b97a90563165d2cf79efd.enc
1c12c5bb95b1d191779ef160624a622a  ./pytorch_model-00001-of-00002.bin.3b0666c50d7fd55d5116e788ec51aa96a34ba6816e86ffbee1dbe983bf511b4b.enc
1a67804dbdfd2168ef30ec077b73e90d  ./pytorch_model-00002-of-00002.bin.763b336a89ef37327716d9c097835720662da656bdc27afde27daec9d0873284.enc
0d6db7f247a51589f3dd6d08dbfe64ce  ./pytorch_model.bin.index.json.4f08b269e18619675bc3fd62f6efb3a8d59f9d54fa50f5625d0bba7adabaf90e.enc
34696bfce7b27548cfc2410e2b55762e  ./special_tokens_map.json.96bdbb8504d9967606e5f661ccc7cbbac44a3661af863a7a58614670a0ccab33.enc
6014cf2235521f974c8d9fb69b6cf07e  ./tokenizer_config.json.7078cc180b3d35e7ccd06b49ede4a7fef85f2572bda40c1fe2fc8f9ab25418d3.enc
56724a79091f3d1877cca65c6412d646  ./tokenizer.model.0b716a618c9e7c45648f91d997431eba3b0ff111b17ce7b777280ed771a49f95.enc
```

## 模型合并
有了原版LLaMA模型以及增量模型之后，就可以进行合并，得到完整的预训练模型。

使用下面的脚本进行模型合并。
其中`/path/to_encrypted`表示刚刚下载的增量模型的路径, 
`/path/to_original_llama_7B`表示原版LLaMA模型的路径，
`/path/to_finetuned_model`表示合并后的模型的路径。

```bash
mkdir /path/to_finetuned_model
for f in "/path/to_encrypted"/*; \
    do if [ -f "$f" ]; then \
       python3 decrypt.py "$f" "/path/to_original_llama_7B/consolidated.00.pth" "/path/to_finetuned_model/"; \
    fi; \
done
```

运行完上面的脚本之后，得到如下文件：

```
./config.json
./generation_config.json
./pytorch_model-00001-of-00002.bin
./pytorch_model-00002-of-00002.bin
./pytorch_model.bin.index.json
./special_tokens_map.json
./tokenizer_config.json
./tokenizer.model
```

## 检查md5sum

```
md5sum ./*
df363050c4ded5c3136270cef715a7d1  ./config.json
2917a1cafb895cf57e746cfd7696bfe5  ./generation_config.json
a88865ce42f45c0c88cd4f7f8ecd75ea  ./pytorch_model-00001-of-00002.bin
ce23ee57ecc73a78b0117e38a68f8d84  ./pytorch_model-00002-of-00002.bin
e5385004e4876ea6b93d6126e845a82f  ./pytorch_model.bin.index.json
15f7a943faa91a794f38dd81a212cb01  ./special_tokens_map.json
08f6f621dba90b2a23c6f9f7af974621  ./tokenizer_config.json
6ffe559392973a92ea28032add2a8494  ./tokenizer.model
```

通过以上步骤，就得到了完整的预训练模型。


# 指令精调
为了让预训练模型更好地跟随用户指令，需要对预训练模型进行指令微调。
传统上，对模型进行微调的方式是通过反向传播算法，将所有的参数进行更新，因此也被称为全量微调。

## 训练脚本
训练的启动脚本在`scripts`目录下，支持单机多卡和多机多卡训练。

### 单机多卡训练

```bash
bash scripts/run.sh
```

- model_name_or_path 代表预训练模型（如果是LLaMA模型，需事先转为hf格式才能通过from_pretrained读取）
- train_file 代表训练数据
- validation_file 代表验证数据
- output_dir 代表训练日志和模型保存的路径
- cache_dir 代表缓存数据处理过程的路径
- cutoff_len 代表最长输入序列长度（LLaMA模型建议设置为1024以上，Bloom模型设置为512以上）

run.sh中包含了全量参数微调和LoRA两种训练方式的启动命令。
下面的命令是单机多卡进行全量参数微调，同时采用deepspeed，基础模型是LLaMA

```bash
torchrun --nproc_per_node 8 train.py \
    --model_name_or_path ${model_name_or_path} \
    --llama \
    --deepspeed configs/deepspeed_config.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 2 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --learning_rate 8e-6 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --fp16 True \
    --seed 1234 \
    --gradient_checkpointing True \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir}
```

**参数说明**

1. 如果想要单卡训练，仅需将nproc_per_node设置为1即可
2. 如果预训练模型不是LLaMA，则去掉--llama。如果是LLaMA模型，需要指定--llama。因为LLaMA模型需要采用LLamaTokenizer加载，如果用AutoTokenizer加载llama可能会出现无限递归的问题，这和transformers版本有关
3. 如果运行环境不支持deepspeed，去掉--deepspeed
4. 全量微调至少需要使用一张A100 80G的显卡。下表列出了实验中使用的超参数。

| 超参数 | 值 |
| --- | --- |
| 精度（Precision） | fp16 |
| 轮数（Epochs） | 2 |
| 批大小（Batch size） | 2 |
| 学习率（Learning rate） | 8e-6 |
| 权值衰减（weight decay） | 0.00001 |
| 学习率预热比例（Warmup ratio） | 0.05 |
| 学习率调度器类型（LR scheduler type） | cosine |

### 多机多卡训练

以两台机器为例，每台机器上有8张卡

首先需要在第一台机器(主机器)上运行

```bash
bash scripts/multinode_run.sh 0
```

然后在第二台机器上运行

```bash
bash scripts/multinode_run.sh 1
```

**参数说明**

```bash
node_rank=$1
echo ${node_rank}
master_addr="10.111.112.223"

# #Multi-node
torchrun --nproc_per_node 8 --nnodes 2 --master_addr ${master_addr} --master_port 14545 --node_rank ${node_rank} src/train.py 
```

- node_rank 代表节点的rank，第一台机器（主机器）的rank设置为0，第二台机器的rank设置为1
- nnodes 代表节点机器的数量
- master_addr 代表主机器的ip地址
- master_port 代表与主机器通信的端口号

## Deepspeed

[DeepSpeed](https://github.com/microsoft/DeepSpeed)是微软推出的大规模分布式训练工具，实现了[ZeRO](https://arxiv.org/abs/1910.02054)论文中描述的所有内容。目前，它全面支持：

1. 优化器状态切分（ZeRO stage 1）
2. 梯度切分（ZeRO stage 2）
3. 参数切分（ZeRO stage 3）
4. 自定义混合精度训练处理
5. 一系列fast CUDA-extension-based optimizers
6. ZeRO-Offload到CPU和磁盘/NVMe

注：如果显存充足，可优先考虑stage 2，对应的配置文件是configs/deepspeed_config.json。如果显存不足，可采用stage 3，该模式采用模型参数并行，可显著减小显存占用，对应的配置文件是configs/deepspeed_config_stage3.json。（在stage=3 模式下，默认不会保存模型的权重，要指定stage3_gather_16bit_weights_on_model_save 为True）

训练日志和模型保存在output_dir目录下，目录下的文件结构应该如下：

```Arduino
output_dir/
├── checkpoint-244/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── trainer_state.json
├── checkpoint-527/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── trainer_state.json
├── pytorch_model.bin
├── print_log.txt
└── config.json
```

trainer_state.json记录了loss、learning_rate的变化

deepspeed 的参数配置可参考：

1. https://www.deepspeed.ai/docs/config-json/
2. https://huggingface.co/docs/accelerate/usage_guides/deepspeed
3. https://github.com/huggingface/transformers/blob/main/tests/deepspeed


## 损失函数

全量微调使用交叉熵（Cross Entropy）损失函数，用来衡量模型生成的文本和人类反馈的文本之间的差异。交叉熵损失函数的公式如下：

$$
{\cal L}(\theta)=-\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{\vert y_{i}\vert}\log P(y_{i j}\vert x_{i};\theta)
$$

其中，$\theta$是模型的参数，$N$是训练集的样本量，$x_i$是第$i$个样本的输入，$\vert y_i \vert$是第$i$个样本的金标输出的长度，$P(y_{ij} ∣ x_i;\theta)$是模型预测第$i$个样本中第$j$个词的概率。

交叉熵损失函数可以反映模型生成的文本和标注文本之间的相似度，越小表示越相似，越大表示越不相似。模型的目标是通过优化参数$\theta$来最小化损失函数，从而提高生成文本的质量。


## 优化器
DeepSpeed原生支持**Adam**、**AdamW**、**OneBitAdam**、**Lamb**和**OneBitLamb**

本项目采用了AdamW优化器，它是 `Adam` 优化器的一种变体。它的作用是基于梯度更新神经网络的参数，使得损失函数最小化。

AdamW 的名称来自于两个部分：`Adam` 和 `Weight Decay`。其中，`Adam` 优化器是一种基于梯度的优化方法，它可以自适应地调整每个参数的学习率，同时具有较好的收敛性能和鲁棒性。`Weight Decay` 是正则化的一种形式，用于防止过拟合，它通过对权重进行衰减来限制模型复杂度。

相比于标准的 `Adam` 优化器，`AdamW` 引入了一个额外的权重衰减项，它在每次参数更新时对权重进行衰减，从而更加有效地控制模型的复杂度。这个权重衰减项的形式与标准的 L2 正则化类似，但是它被证明在某些情况下可以更好地控制模型的过拟合。
