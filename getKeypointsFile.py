def getKeypoints(path):
    listas = []
    with open(path, 'r') as file:
        lines = file.readlines()
        
        list_person = []
        for line in lines:
            line = line.strip()

            if len(line) == 2:
                break
            if len(line) < 21:
                listas.append(list_person)
                list_person = []
            else:
                line = line.strip()
                line = line.split('[')[-1].strip()
                line = line.split("]")[0].strip()
                line = line.split(' ')

                # adaptar al derecho
                try:
                    x = float(line[0])
                except:
                    x = float(line[0].split(",")[0])
                y = float(line[-1])
                list_person.append([x, y])
        
        if len(list_person) > 0:
            listas.append(list_person)

    return listas