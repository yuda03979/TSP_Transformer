from itertools import takewhile


class Tokenizer:

    def __init__(self, special_tokens: list, vocab_size: int, board_size: tuple[int, int], seq_len: int):
        self.special_tokens = special_tokens
        self.vocab_size = vocab_size
        self.board_size = board_size
        self.seq_len = seq_len
        self.points_len = (seq_len // 2) + 1  # seq_len already include special tokens

        self.mask_token = self.special_tokens[0]
        self.sop_token = self.special_tokens[1]
        self.eop_token = self.special_tokens[2]

        self.mask_list = [self.mask_token for _ in range(seq_len)]

    def fill_mask(self, points, until):
        return (points + self.mask_list)[:until]

    def mapping(self, token: tuple | int | str, reverse: bool = False) -> tuple | str | int:
        if not reverse:
            if isinstance(token, tuple):
                if not (0 <= token[0] < self.board_size[0] and 0 <= token[1] < self.board_size[1]):
                    raise ValueError(f"Coordinates {token} out of bounds")
                return (self.board_size[1] * token[1]) + token[0]
            elif token in self.special_tokens:
                base_tokens = self.board_size[0] * self.board_size[1]
                return base_tokens + self.special_tokens.index(token)
            else:
                raise ValueError(f"Invalid token: {token}")
        else:
            board_positions = self.board_size[0] * self.board_size[1]

            if token < board_positions:
                x = token % self.board_size[1]
                y = token // self.board_size[1]
                return x, y
            elif token < self.vocab_size:
                special_idx = token - board_positions
                return self.special_tokens[special_idx]
            else:
                raise ValueError(f"Index {token} out of vocabulary range")

    def tokenize_points(self, points: list[tuple]) -> list:

        points = [self.sop_token] + points
        points = self.fill_mask(points, self.points_len - 1)
        points.append(self.eop_token)

        return [self.mapping(p) for p in points]

    def detokenize_points(self, points: list[int]) -> list:
        points = [self.mapping(t, reverse=True) for t in points]
        points = [p for p in points if isinstance(p, tuple)]
        return points

    def tokenize_path(self, points: list[tuple]) -> list:

        points = self.fill_mask(points, self.points_len - 2)
        return [self.mapping(p) for p in points]

    def detokenize_path(self, points: list[int]):
        points = [self.mapping(t, reverse=True) for t in points]
        return list(takewhile(lambda p: p not in self.special_tokens, points))  # [start:spacial_token]

    def tokenize_token(self, token: tuple) -> int:
        return self.mapping(token)

    def detokenize_token(self, token: int) -> tuple | str:
        return self.mapping(token, reverse=True)

