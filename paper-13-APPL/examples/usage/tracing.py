from concurrent.futures import ThreadPoolExecutor

from appl import gen, ppl, print_trace, traceable


@ppl
def add(x: int, y: int):
    f"{x}+{y}=?"
    return gen()


@traceable
def calc():
    return [add(x, y) for x in range(3) for y in range(3)]


@traceable
def calc_with_thread():
    with ThreadPoolExecutor(max_workers=5) as executor:
        return list(executor.map(add, range(9), range(9)))


@traceable
def main():
    print("with APPL asynchronous generation")
    for res in calc():
        print(res)
    print("With ThreadPoolExecutor")
    for res in calc_with_thread():
        print(res)


if __name__ == "__main__":
    main()
    print_trace()  # print the trace, default is to langfuse.
