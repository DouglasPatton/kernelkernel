from pi_db_tool import DBTool
if __name__ == "__main__":
    dbt=DBTool()
    results_dict=dbt.resultsDBdict()
    gen_dict=dbt.genDBdict()
    print('len(results_dict): ',len(results_dict))
    print('len(gen_dict) ',len(gen_dict))
