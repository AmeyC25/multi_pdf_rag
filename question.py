def count_valid_brackets(s):
    stack = []
    count = 0
    for char in s:
        if char == '[':
            stack.append(char)
        elif char == ']':
            if stack:
                stack.pop()
                count += 1
    return count

s = "]][][[]][[[[[][[["
print(count_valid_brackets(s))
