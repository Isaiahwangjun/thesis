import sys

with open(sys.argv[1]) as fh:
    sum = 0 # initialize here, outside the loop
    count = 0 # and a line counter
    for line in fh:
        count += 1 # increment the counter
        sum += float(line) # add here, not in a nested loop
    average = sum / count

print(average)
