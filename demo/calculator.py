"""演示用计算器模块。

注意：multiply 中故意植入了一个 bug（用了 + 而非 *），
用于演示 Agent 如何定位并修复测试失败。
"""


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a + b  # BUG: 应为 a * b


def divide(a, b):
    if b == 0:
        raise ValueError("division by zero")
    return a / b
