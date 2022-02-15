with open('cutoff_frequencies.txt') as f:
    lines = f.readlines()

for l in lines:
    print(int(l))