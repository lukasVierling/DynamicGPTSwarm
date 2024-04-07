import os
from datasets import load_dataset

def download():
    this_file_path = os.path.split(__file__)[0]
    data_path = os.path.join(this_file_path, "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    task_list = ['agronomy', 'anatomy', 'ancient_chinese', 'arts', 'astronomy', 'business_ethics', 'chinese_civil_service_exam', 'chinese_driving_rule', 'chinese_food_culture', 'chinese_foreign_policy', 'chinese_history', 'chinese_literature', 
    'chinese_teacher_qualification', 'clinical_knowledge', 'college_actuarial_science', 'college_education', 'college_engineering_hydrology', 'college_law', 'college_mathematics', 'college_medical_statistics', 'college_medicine', 'computer_science',
    'computer_security', 'conceptual_physics', 'construction_project_management', 'economics', 'education', 'electrical_engineering', 'elementary_chinese', 'elementary_commonsense', 'elementary_information_and_technology', 'elementary_mathematics', 
    'ethnology', 'food_science', 'genetics', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_geography', 'high_school_mathematics', 'high_school_physics', 'high_school_politics', 'human_sexuality',
    'international_law', 'journalism', 'jurisprudence', 'legal_and_moral_basis', 'logical', 'machine_learning', 'management', 'marketing', 'marxist_theory', 'modern_chinese', 'nutrition', 'philosophy', 'professional_accounting', 'professional_law', 
    'professional_medicine', 'professional_psychology', 'public_relations', 'security_study', 'sociology', 'sports_science', 'traditional_chinese_medicine', 'virology', 'world_history', 'world_religions']

    for task in task_list:
        for split in ['dev', 'test']:
            split_path = os.path.join(data_path, split, task)
            if not os.path.exists(split_path):
                os.makedirs(split_path)
            csv_path = os.path.join(split_path, f"{task}.csv")
            if os.path.exists(csv_path):
                #print(f"Dataset {task} already downloaded. Skipping.")
                continue
            dataset = load_dataset("haonan-li/cmmlu", task)
            dataset[split].to_csv(csv_path)
            print(f"Saved to {csv_path}")

if __name__ == "__main__":
    download()