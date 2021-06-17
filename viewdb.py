import os
from pi_db_tool import DBTool
if __name__ == "__main__":
    dbt=DBTool()
    db_dict={
        'resultsDBdict':dbt.resultsDBdict(),
        'gen_dict':dbt.genDBdict(),
        'fitfail_dict':dbt.fitfailDBdict()}
    if os.path.exists(dbt.predictDBdictpath):
        db_dict['predictDBdict']=dbt.predictDBdict()
    
    for db_name,db_callable in db_dict.items():
        with db_callable as db:
            print(f'len({db_name}): {len(db)}')
        
        
    #        success=0
    #        fail=0
    '''
    for key in results_dict.keys():
        try:
            r=results_dict[key]
            success+=1
        except:
            fail+=1
    print('resultsdict:')
    print('successes: ',success)
    print('failures: ',fail)
    success=0
    fail=0
    for key in gen_dict.keys():
        try:
            r=gen_dict[key]
            success+=1
        except:
            fail+=1
    print('gendict:')
    print('successes: ',success)
    print('failures: ',fail)

    '''
