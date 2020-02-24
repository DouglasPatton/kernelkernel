if __name__=="__main__":
    from multicluster import mypool
    import psutil
    cores=int(psutil.cpu_count(logical=True))
    print(f'choose a node count between 1 and {cores-2}')
    nodecount=int(input('node count:'))

    mypool(nodecount=nodecount,
                source='pisces',
                includemaster=0,
                local_run=0
               )
