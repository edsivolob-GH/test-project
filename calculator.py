import math

# --- Binary operators ---

def add(a, b):      return a + b
def subtract(a, b): return a - b
def multiply(a, b): return a * b
def modulo(a, b):   return a % b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def power(a, b):    return a ** b

def log_base(a, b):
    if a <= 0:
        raise ValueError("Logarithm argument must be positive")
    if b <= 0 or b == 1:
        raise ValueError("Logarithm base must be positive and not 1")
    return math.log(a, b)

def nth_root(a, b):
    if b == 0:
        raise ValueError("Root degree cannot be zero")
    if a < 0 and b % 2 == 0:
        raise ValueError("Even root of a negative number is undefined")
    return math.copysign(abs(a) ** (1 / b), a)

BINARY_OPS: dict[str, tuple[callable, str]] = {
    "+":    (add,      "<a> + <b>"),
    "-":    (subtract, "<a> - <b>"),
    "*":    (multiply, "<a> * <b>"),
    "/":    (divide,   "<a> / <b>"),
    "%":    (modulo,   "<a> % <b>          modulo"),
    "**":   (power,    "<a> ** <b>         a to the power of b"),
    "log":  (log_base, "<a> log <b>        log base b of a"),
    "root": (nth_root, "<a> root <b>       nth root of a"),
}

# --- Unary functions ---

def _safe_asin(x):
    if not -1 <= x <= 1:
        raise ValueError("asin domain is [-1, 1]")
    return math.degrees(math.asin(x))

def _safe_acos(x):
    if not -1 <= x <= 1:
        raise ValueError("acos domain is [-1, 1]")
    return math.degrees(math.acos(x))

def _safe_sqrt(x):
    if x < 0:
        raise ValueError("Square root of a negative number is undefined")
    return math.sqrt(x)

def _safe_log(x):
    if x <= 0:
        raise ValueError("ln argument must be positive")
    return math.log(x)

def _safe_log10(x):
    if x <= 0:
        raise ValueError("log10 argument must be positive")
    return math.log10(x)

def _safe_log2(x):
    if x <= 0:
        raise ValueError("log2 argument must be positive")
    return math.log2(x)

UNARY_FNS: dict[str, tuple[callable, str]] = {
    "sin":   (lambda x: math.sin(math.radians(x)),  "sin <deg>"),
    "cos":   (lambda x: math.cos(math.radians(x)),  "cos <deg>"),
    "tan":   (lambda x: math.tan(math.radians(x)),  "tan <deg>"),
    "asin":  (_safe_asin,                            "asin <x>           result in degrees"),
    "acos":  (_safe_acos,                            "acos <x>           result in degrees"),
    "atan":  (lambda x: math.degrees(math.atan(x)), "atan <x>           result in degrees"),
    "sqrt":  (_safe_sqrt,                            "sqrt <x>"),
    "cbrt":  (lambda x: math.copysign(abs(x)**(1/3), x), "cbrt <x>"),
    "abs":   (abs,                                   "abs <x>"),
    "ceil":  (math.ceil,                             "ceil <x>"),
    "floor": (math.floor,                            "floor <x>"),
    "round": (round,                                 "round <x>"),
    "fact":  (lambda x: float(math.factorial(int(x))), "fact <n>           factorial"),
    "ln":    (_safe_log,                             "ln <x>             natural log"),
    "log10": (_safe_log10,                           "log10 <x>"),
    "log2":  (_safe_log2,                            "log2 <x>"),
    "exp":   (math.exp,                              "exp <x>            e^x"),
    "deg":   (math.degrees,                          "deg <rad>          radians → degrees"),
    "rad":   (math.radians,                          "rad <deg>          degrees → radians"),
    "sinh":  (math.sinh,                             "sinh <x>"),
    "cosh":  (math.cosh,                             "cosh <x>"),
    "tanh":  (math.tanh,                             "tanh <x>"),
}

CONSTANTS: dict[str, float] = {
    "pi":  math.pi,
    "e":   math.e,
    "tau": math.tau,
    "inf": math.inf,
    "phi": (1 + math.sqrt(5)) / 2,
}

def parse_number(token: str) -> float:
    if token in CONSTANTS:
        return CONSTANTS[token]
    try:
        return float(token)
    except ValueError:
        raise ValueError(f"Unknown number or constant: '{token}'")

class _Parser:
    """
    Recursive descent parser for space-separated infix expressions.

    Precedence (low → high):
      additive      : + -
      multiplicative: * / % log root
      exponent      : ** (right-associative)
      unary         : sin cos sqrt ... (prefix functions, chainable)
      primary       : number | constant | ( expr )
    """

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self) -> str:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def expect(self, value: str) -> None:
        token = self.consume()
        if token != value:
            raise ValueError(f"Expected '{value}', got '{token}'")

    def parse(self) -> float:
        result = self._additive()
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token: '{self.peek()}'")
        return result

    def _additive(self) -> float:
        left = self._multiplicative()
        while self.peek() in ("+", "-"):
            op = self.consume()
            right = self._multiplicative()
            left = BINARY_OPS[op][0](left, right)
        return left

    def _multiplicative(self) -> float:
        left = self._exponent()
        while self.peek() in ("*", "/", "%", "log", "root"):
            op = self.consume()
            right = self._exponent()
            left = BINARY_OPS[op][0](left, right)
        return left

    def _exponent(self) -> float:
        base = self._unary()
        if self.peek() == "**":
            self.consume()
            exp = self._exponent()  # recurse for right-associativity
            return BINARY_OPS["**"][0](base, exp)
        return base

    def _unary(self) -> float:
        if self.peek() in UNARY_FNS:
            fn = self.consume()
            operand = self._unary()  # chainable: sin cos 90
            return UNARY_FNS[fn][0](operand)
        return self._primary()

    def _primary(self) -> float:
        token = self.peek()
        if token is None:
            raise ValueError("Unexpected end of expression")
        if token == "(":
            self.consume()
            result = self._additive()
            self.expect(")")
            return result
        self.consume()
        return parse_number(token)


def calculate(expression: str) -> float:
    tokens = expression.split()
    if not tokens:
        raise ValueError("Empty expression")
    return _Parser(tokens).parse()

def print_help():
    print("\nBinary operators:  <a> <op> <b>")
    for op, (_, desc) in BINARY_OPS.items():
        print(f"  {desc}")
    print("\nUnary functions:   <fn> <x>")
    for _, (_, desc) in UNARY_FNS.items():
        print(f"  {desc}")
    print("\nConstants (usable anywhere a number is expected):")
    print(f"  {', '.join(CONSTANTS)}")
    print()

def main():
    print("Scientific Calculator  (type 'help' or 'quit')")
    print("-" * 48)

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if user_input.lower() in ("help", "h", "?"):
            print_help()
            continue

        try:
            result = calculate(user_input)
            print(f"= {result:g}")
        except (ValueError, OverflowError, ZeroDivisionError) as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
