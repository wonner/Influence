from filecmp import cmp

file = open('data/slashdot.txt')
writefile = open('data/slashdot1.txt','w')

while 1:
    line = file.readline()
    if not line:
        break
    split = line.split('\t')
    if split[0].strip() != split[1].strip():
        writefile.write(line)
file.close()
writefile.close()
