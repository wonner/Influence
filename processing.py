
def delete_selfloop():
    file = open('data/soc-Slashdot0811.txt')
    writefile = open('data/slashdot.txt','w')

    while 1:
        line = file.readline()
        if not line:
            break
        split = line.split('\t')
        if split[0].strip() != split[1].strip():
            writefile.write(split[0]+' '+split[1])
    file.close()
    writefile.close()

def reverse():
    file = open('data/slashdot.txt')
    writefile = open('data/slashdot-reverse.txt','w')

    while 1:
        line = file.readline()
        if not line:
            break
        split = line.split(' ')
        writefile.write(split[1].strip() + " " + split[0]+"\n")
    file.close()
    writefile.close()

reverse()
