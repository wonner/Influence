

file = open('data/karate.edgelist')
writefile = open('data/karate.txt','w')

while 1:
    line = file.readline()
    if not line:
        break
    split = line.split(' ')
    writefile.write(str(int(split[0])-1) + " " + str(int(split[1])-1)+"\n")
file.close()
writefile.close()
