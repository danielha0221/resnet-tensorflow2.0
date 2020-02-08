import resnet

def main():
  net = resnet.ResNet(34)

  net.create_model()
  net.train()

if __name__ == '__main__':
    main()