import os


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path_dict = {
        "PCG": os.path.join(
            dir_path,
            "../data/",
            "250528_PCG_assessment_report_Q1-2025_controlperrow.xlsx",
        ),
        "risk_catalog-20250508": os.path.join(
            dir_path,
            "../data/",
            "RMI-V2-Translate_20250508.xlsx",
        ),
        "lotus_south": os.path.join(
            dir_path,
            "../data/",
            "250520_Retail (Guangdong-Guangxi)_assessment_report_Q1-2025_controlperrow_original.xlsx",
        ),
        "risk_catalog-20250327": os.path.join(
            dir_path,
            "../data/",
            "250327_data_from_RMI_from_productionbuild_Q3-2024_label 1.xlsx",
        ),
    }
