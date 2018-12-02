import os
path = "C:\\raj\\imag1"

i = 0

print path

for dirname , dirnames , filenames in os.walk ( path ):
        for file_ in filenames :
            naam=os.path.join(dirname, file_);
            print(naam)
	    print(dirname)
	    a,b,c,d=dirname.split("\\")
	    print(c)
	    print(d)
	    print(file_)
	    