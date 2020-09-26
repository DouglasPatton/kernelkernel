from pi_db_tool import DBTool
if __name__ == "__main__":
    dbt=DBTool()
    results_dict=dbt.resultsDBdict()
    gen_dict=dbt.genDBdict()
    print('len(results_dict): ',len(results_dict))
    print('len(gen_dict) ',len(gen_dict))
    success=0
    fail=0
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