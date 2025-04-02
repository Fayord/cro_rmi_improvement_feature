import pandas as pd
from datetime import datetime


def convert_feedback_to_excel(feedback_data, output_file=None):
    """Convert feedback data to Excel format with multiple sheets."""

    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"risk_feedback_review_{timestamp}.xlsx"

    # Create Excel writer
    with pd.ExcelWriter(output_file) as writer:
        # Original Text and Metrics Sheet
        original_data = {
            "Original Text": [item["consolidated_feedback"]["original_text"] for item in feedback_data],
            "Invalid Metrics": [", ".join(item["consolidated_feedback"]["invalid_metrics"]) for item in feedback_data],
            "Feedback Message": [item["consolidated_feedback"]["question"] for item in feedback_data]
        }
        pd.DataFrame(original_data).to_excel(
            writer, sheet_name="Original Analysis", index=False
        )

        # Examples Sheet
        examples_rows = []
        for idx, item in enumerate(feedback_data, 1):
            examples = item["consolidated_feedback"]["examples"]
            for ex_num, choice in enumerate(["choice1", "choice2", "choice3"], 1):
                examples_rows.append({
                    "Feedback Number": f"Feedback {idx}",
                    "Example Number": f"Example {ex_num}",
                    "Improved Text": examples[choice],
                    "Reviewer Comments": "",
                    "Approved": ""
                })
        
        pd.DataFrame(examples_rows).to_excel(
            writer, sheet_name="Examples Review", index=False
        )

        # Review Template Sheet
        review_rows = []
        for idx, _ in enumerate(feedback_data, 1):
            for aspect in ["Clarity", "Completeness", "Accuracy", "Overall Quality"]:
                review_rows.append({
                    "Feedback Number": f"Feedback {idx}",
                    "Aspect": aspect,
                    "Rating (1-5)": "",
                    "Comments": ""
                })
        
        pd.DataFrame(review_rows).to_excel(
            writer, sheet_name="Quality Review", index=False
        )

    return output_file


# Example usage
feedback_data = [
    {
        "consolidated_feedback": {
            "invalid_metrics": ["Have cause", "Have impact"],
            "question": "ข้อความตอบกลับ: \n\nเพื่อปรับปรุงคำอธิบายความเสี่ยงของคุณ ควรเพิ่มการวิเคราะห์เกี่ยวกับสาเหตุที่ทำให้เกิดการทุจริตในกระบวนการขายและการเบิกจ่ายของคลังสินค้า เช่น ปัจจัยที่กระตุ้นหรือสภาพแวดล้อมที่เอื้อต่อการทุจริต นอกจากนี้ ควรอธิบายผลกระทบที่อาจเกิดขึ้นจากความเสี่ยงเหล่านี้ เช่น การสูญเสียทางการเงิน ความเสียหายต่อชื่อเสียง หรือการหยุดชะงักในการดำเนินงาน การเพิ่มข้อมูลเหล่านี้จะช่วยให้คำอธิบายความเสี่ยงมีความชัดเจนและครอบคลุมมากขึ้น ซึ่งจะเป็นประโยชน์ต่อการประเมินความเสี่ยงในอนาคต.",
            "examples": {
                "choice1": "1. การทุจริตในกระบวนการขายและการเบิกจ่ายของส่งเสริมการขายเกิดจากการขาดการตรวจสอบภายในที่เข้มงวด ซึ่งอาจนำไปสู่การสูญเสียทางการเงินและความเชื่อมั่นของลูกค้า 2. การทุจริตในกระบวนการเบิกจ่ายของคลังสินค้าสืบเนื่องจากการขาดการควบคุมที่มีประสิทธิภาพ ส่งผลให้เกิดการขาดแคลนสินค้าหรือการจัดการสินค้าที่ไม่ถูกต้อง",
                "choice2": "1. การทุจริตในกระบวนการขายและการเบิกจ่ายของส่งเสริมการขายมีสาเหตุมาจากการขาดการฝึกอบรมพนักงานที่เพียงพอ ซึ่งอาจส่งผลให้เกิดการสูญเสียรายได้และลดความน่าเชื่อถือของแบรนด์ 2. การทุจริตในกระบวนการเบิกจ่ายของคลังสินค้าสาเหตุจากการขาดระบบติดตามที่มีประสิทธิภาพ ทำให้เกิดความไม่ถูกต้องในการจัดการสินค้าคงคลังและการสูญเสียทรัพยากร",
                "choice3": "1. การทุจริตในกระบวนการขายและการเบิกจ่ายของส่งเสริมการขายเกิดจากการขาดการตรวจสอบและการควบคุมภายในที่เหมาะสม ซึ่งอาจนำไปสู่การสูญเสียทางการเงินและความเสี่ยงต่อการฟ้องร้อง 2. การทุจริตในกระบวนการเบิกจ่ายของคลังสินค้าสืบเนื่องจากการขาดการประเมินความเสี่ยงที่เหมาะสม ส่งผลให้เกิดการจัดการสินค้าที่ไม่ถูกต้องและความเสียหายต่อชื่อเสียงขององค์กร",
            },
            "original_text": "1.การทุจริตในกระบวนการขาย การเบิกจ่าย ของส่งเสริมการขาย 2.การทุจริตในกระบวนการเบิกจ่ายของคลังสินค้า",
        }
    }
]

output_file = convert_feedback_to_excel(feedback_data)
print(f"Excel file created: {output_file}")
