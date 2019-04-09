from libskynet import*
import json

def main():

    ### Create net
    net = snResNet50.createNet(snType.calcMode.CPU)

    ### Set weight
    weightTF = snWeight.getResNet50Weights()

    if (not snResNet50.setWeights(net, weightTF)):
        print('Error setWeights')
        exit(-1)

    arch = json.loads(net.getGetArchitecNet())

    with open('resNet50Struct.json', 'w', encoding='utf-8') as outfile:
        json.dump(arch, outfile, sort_keys=True, indent=4, ensure_ascii=False)

    if (not net.saveAllWeightToFile("resNet50Weights.dat")):
        print('Error saveAllWeightToFile')
        exit(-1)

    return 0

if __name__ == '__main__':
    main()



