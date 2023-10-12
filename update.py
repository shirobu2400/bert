import torch
import time
import random
import bot
import tqdm
import argparse

# 文字単位での計算 -> False の場合形態素解析を行う
is_moji: bool = True


def create_batch(datas: list, size: int):
    return [datas[random.randrange(len(datas))] for _ in range(size)]


def create_batch_pair_random(datas1: list, datas2: list, size: int):
    r1 = []
    r2 = []
    for _ in range(size):
        i = random.randrange(len(datas1))
        r1.append(datas1[i])
        r2.append(datas2[i])
    return r1, r2


def create_batch_pair_index(datas1: list, datas2: list, size: int):
    r1 = []
    r2 = []
    for i in range(size):
        r1.append(datas1[i % len(datas1)])
        r2.append(datas2[i % len(datas2)])
    return r1, r2


def test(b: bot.bot, log_name: str, batch_size: int = 256):
    with open(log_name, encoding="utf-8_sig") as fp:
        text: list[str] = fp.readlines()
    text = list(set(text))
    for i in range(0, len(text), batch_size):
        inputs_t, _ = create_batch_pair_index(text[i: i + batch_size], text[i: i + batch_size], batch_size)

        t = inputs_t
        x: torch.Tensor = b.convert_itot(b.convert_stoi(inputs_t))

        # t = t.replace("\n", "") + "\0"
        p = b.forward(x, 1).cpu()
        a = b.convert_itos([x.tolist() for x in p])
        for tt, aa in zip(t, a):
            print(tt.replace("\n", ""), "<==>", aa)


def train(bot_name: str, log_name: str, iter_size: int, is_continue: bool, epoch_size: int = 100, batch_size: int = 256):

    # df = pd.read_csv("Bokuraga_son.csv")
    # text = df["text"].values.tolist()

    with open(log_name, encoding="utf-8_sig") as fp:
        text: list[str] = fp.readlines()

    # inputs から teachs を推定させる
    step_size: int = 1
    inputs: list[str] = [t.replace("\n", "") + "\0" for t in text[:-1:step_size]]
    teaches: list[str] = [t.replace("\n", "") + "\0" for t in text[1::step_size]]

    if iter_size == 0:
        iter_size = len(text)

    # モデルの定義
    b = bot.bot(20)
    if is_continue:
        try:
            b.load(bot_name)
        except FileNotFoundError:
            print(bot_name, "not found.")
    if not is_moji:
        b.using_morphological_analysis(inputs + teaches)

    # 実行時間の計測開始
    start_time: float = time.time()

    # epoch_size 回の訓練実行
    with tqdm.tqdm(range(epoch_size * iter_size)) as progress:
        for itr in progress:
            epoch = itr // iter_size

            inputs_t, teaches_t = create_batch_pair_random(inputs, teaches, batch_size)

            ptv_inputs: torch.Tensor = b.convert_itot(b.convert_stoi(inputs_t))
            ptv_teachs: torch.Tensor = b.convert_itot(b.convert_stoi(teaches_t))

            # Parameter Update.
            loss = 0.00
            loss = b.update(ptv_inputs, ptv_teachs)

            progress.set_description("epoch {0}, loss: {1}".format(epoch, round(loss, 6)))

    print("Finished.")

    # 経過時間の表示
    print("Elapsed time is {0}".format(time.time() - start_time))

    # 保存
    b.save(bot_name)
    return b


# 実行
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="transformers program.",
        description="transformers updater test.")
    parser.add_argument("-b", "--bot", type=str, default="memory/bot.pkl")
    parser.add_argument("-f", type=str, default="logs/rev.log")
    parser.add_argument("-i", type=int, default=0)
    parser.add_argument("-e", type=int, default=100)
    parser.add_argument("-c", "--is_continue", action="store_true")
    args = parser.parse_args()

    b = train(args.bot, args.f, args.i, args.is_continue, args.e)
    test(b, args.f)
