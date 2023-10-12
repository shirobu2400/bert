import joyo


class moji:
    def __init__(self):
        mojis: list[str] = ["", ""] \
            + self.alphabet_lower() + self.alphabet_upper() \
            + self.hiragana() + self.katakana() \
            + self.zenkaku_num() + self.hankaku() \
            + [x for x in joyo.load()]
        mojis = self.check_sjisutf8(mojis)

        self.moji2code_cmap: dict[str, int] = {}
        self.code2moji_cmap: dict[int, str] = {}
        for i, c in enumerate(mojis):
            self.moji2code_cmap[c] = i
            self.code2moji_cmap[i] = c

    def code2moji(self, code: list[int]) -> str:
        mojis: str = ""
        for c in code:
            if c in self.code2moji_cmap.keys():
                mojis = mojis + self.code2moji_cmap[c]
        return mojis

    def moji2code(self, mojis: str, maxrange: int = 4) -> list[int]:
        codes: list[int] = []
        while len(mojis) > 0:
            for j in range(min(maxrange, len(mojis)), 0, -1):
                c = mojis[: j]
                if c in self.moji2code_cmap.keys():
                    codes.append(self.moji2code_cmap[c])
                    mojis = mojis[j - 1:]
                    break
            mojis = mojis[1:]
        return codes

    def size(self):
        return len(self.moji2code_cmap.keys())

    # 文字リスト
    # https://qiita.com/okkn/items/3aef4458ed2269a59d63
    @staticmethod
    def alphabet_lower() -> list[str]:
        return [chr(i) for i in range(97, 97+26)]

    @staticmethod
    def alphabet_upper() -> list[str]:
        return [chr(i) for i in range(65, 65+26)]

    @staticmethod
    def hiragana() -> list[str]:
        return [chr(i) for i in range(12353, 12436)]

    @staticmethod
    def katakana() -> list[str]:
        return [chr(i) for i in range(12449, 12532+1)]

    @staticmethod
    def zenkaku_num() -> list[str]:
        return [chr(i) for i in range(65296, 65296+10)]

    @staticmethod
    def hankaku() -> list[str]:
        return [chr(i) for i in range(32, 127)]

    @staticmethod
    def check_sjisutf8(mojis: list[str]) -> list[str]:
        # sjis, utf8 で変換可能であるか
        r: list[str] = []
        for c in mojis:
            try:
                c.encode('utf-8')
                c.encode('sjis')
                r.append(c)
            except UnicodeEncodeError:
                pass
        return r

# if __name__ == "__main__":
#     mojis: moji = moji()
#     codes = mojis.moji2code("こんばんは")
#     print(codes)
#     mojis_rebuild = mojis.code2moji(codes)
#     print(mojis_rebuild)
