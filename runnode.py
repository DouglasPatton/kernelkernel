if __name__=="__main__":
    import mycluster
    import psutil
    cores=int(psutil.cpu_count(logical=False))
    print(f'choose a node count between 1 and {cores-2}')
    nodecount=int(input('node count:'))

    mycluster.mypool(nodecount=nodecount,
                source='pisces',
                includemaster=0,
                local_run=0
               )
