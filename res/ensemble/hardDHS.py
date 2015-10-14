__author__ = 'Fule Liu'

if __name__ == "__main__":
    with open("hardDHS.txt") as fp:
        lines = fp.readlines()

    count = 0
    for line in lines[1:]:
        line = [int(e) for e in line.rstrip().split()]
        if sum(line[2:]) == -4:
            print(line)
            count += 1

    print(count)