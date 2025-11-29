"""
Few-shot examples with Chain-of-Thought reasoning for QA system.
Curated examples covering various question types and domains.
"""

# Example 1: Technical question with clear evidence
EXAMPLE_1 = """
EVIDENCE:
DES (Data Encryption Standard) được phát triển tại IBM vào đầu những năm 1970. DES là dạng mã hóa khối với khối dữ liệu vào kích thước 64 bit và khóa 64 bit, trong đó thực sử dụng 56 bit (còn gọi là kích thước hiệu dụng của khóa) và 8 bit dùng cho kiểm tra chẵn lẻ.

QUESTION: Trong giải thuật DES, kích thước hiệu dụng của khóa là bao nhiêu bit?

OPTIONS:
A. 48 bit
B. 56 bit
C. 64 bit
D. 128 bit

REASONING:
1. Evidence Analysis: Tài liệu nêu rõ DES sử dụng khóa 64 bit, nhưng chỉ thực sử dụng 56 bit (gọi là kích thước hiệu dụng), còn 8 bit dùng cho kiểm tra chẵn lẻ.

2. Option Evaluation:
   - Option A (48 bit): Không được đề cập trong tài liệu, loại bỏ.
   - Option B (56 bit): Khớp chính xác với "thực sử dụng 56 bit (còn gọi là kích thước hiệu dụng của khóa)" - Đúng.
   - Option C (64 bit): Đây là tổng kích thước khóa, không phải kích thước hiệu dụng, loại bỏ.
   - Option D (128 bit): Không được đề cập, loại bỏ.

3. Conclusion: Dựa trên bằng chứng rõ ràng trong tài liệu, đáp án là B (56 bit).

ANSWER: {"A": 0.0, "B": 1.0, "C": 0.0, "D": 0.0}
"""

# Example 2: Mathematical calculation question
EXAMPLE_2 = """
EVIDENCE:
Khi thực hiện PCA, tỉ lệ phương sai được giữ lại khi chọn K thành phần chính được tính bằng tổng K trị riêng lớn nhất chia cho tổng tất cả trị riêng. Công thức: (λ1 + λ2 + ... + λK) / (λ1 + λ2 + ... + λn).

QUESTION: Cho ma trận có các trị riêng λ1=5, λ2=2, λ3=1, λ4=0.5 (giảm dần). Nếu chọn K=2 thành phần chính, tỉ lệ phương sai được giữ lại là bao nhiêu?

OPTIONS:
A. 0.824
B. 0.588
C. 1.167
D. 0.235

REASONING:
1. Evidence Analysis: Công thức tính tỉ lệ phương sai = (tổng K trị riêng lớn nhất) / (tổng tất cả trị riêng).

2. Calculation:
   - K = 2, nên lấy 2 trị riêng lớn nhất: λ1=5, λ2=2
   - Tổng K trị riêng: 5 + 2 = 7
   - Tổng tất cả trị riêng: 5 + 2 + 1 + 0.5 = 8.5
   - Tỉ lệ = 7 / 8.5 ≈ 0.824

3. Option Evaluation:
   - Option A (0.824): Khớp với kết quả tính toán - Đúng.
   - Option B (0.588): Sai, có thể nhầm lẫn công thức.
   - Option C (1.167): Lớn hơn 1, không hợp lý cho tỉ lệ phần trăm, loại bỏ.
   - Option D (0.235): Quá nhỏ, loại bỏ.

4. Conclusion: Đáp án là A (0.824).

ANSWER: {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0}
"""

# Example 3: Multi-answer question
EXAMPLE_3 = """
EVIDENCE:
Điều trị áp lực đường thở dương có thể được chống chỉ định ở một số bệnh nhân có các bệnh lý: bệnh kén khí phổi nặng, tràn khí màng phổi, huyết áp thấp bệnh lý, mất nước, rò rỉ dịch não tủy, phẫu thuật sọ não gần đây hoặc chấn thương.

QUESTION: Chọn tất cả các chống chỉ định của điều trị áp lực đường thở dương?

OPTIONS:
A. Bệnh kén khí phổi nặng
B. Huyết áp cao
C. Tràn khí màng phổi
D. Mất nước

REASONING:
1. Evidence Analysis: Tài liệu liệt kê rõ các chống chỉ định gồm: bệnh kén khí phổi nặng, tràn khí màng phổi, huyết áp thấp bệnh lý, mất nước, và một số trường hợp khác.

2. Option Evaluation:
   - Option A (Bệnh kén khí phổi nặng): Có trong danh sách chống chỉ định - Đúng.
   - Option B (Huyết áp cao): Không được đề cập, tài liệu nói "huyết áp thấp" chứ không phải cao - Sai.
   - Option C (Tràn khí màng phổi): Có trong danh sách chống chỉ định - Đúng.
   - Option D (Mất nước): Có trong danh sách chống chỉ định - Đúng.

3. Conclusion: Câu hỏi yêu cầu chọn tất cả, nên đáp án là A, C, D.

ANSWER: {"A": 1.0, "B": 0.0, "C": 1.0, "D": 1.0}
"""

# Example 4: Low evidence scenario
EXAMPLE_4 = """
EVIDENCE:
IoT (Internet of Things) đề cập đến mạng lưới các thiết bị vật lý được kết nối với internet, cho phép thu thập và trao đổi dữ liệu. Các ứng dụng IoT bao gồm nhà thông minh, thành phố thông minh, và y tế từ xa.

QUESTION: Blockchain có thể được sử dụng trong IoT để làm gì?

OPTIONS:
A. Tăng cường bảo mật và minh bạch dữ liệu
B. Giảm băng thông mạng
C. Thay thế hoàn toàn cloud computing
D. Loại bỏ nhu cầu về cảm biến

REASONING:
1. Evidence Analysis: Tài liệu chỉ giới thiệu về IoT, không đề cập trực tiếp đến Blockchain. Tuy nhiên, dựa trên kiến thức chung về công nghệ.

2. Option Evaluation:
   - Option A (Tăng cường bảo mật và minh bạch): Blockchain được biết đến với khả năng bảo mật cao và tính minh bạch, phù hợp với IoT - Có khả năng đúng.
   - Option B (Giảm băng thông): Blockchain thường tốn băng thông do cần đồng bộ, không phải lợi ích chính - Sai.
   - Option C (Thay thế cloud): Blockchain bổ trợ chứ không thay thế hoàn toàn cloud - Sai.
   - Option D (Loại bỏ cảm biến): Blockchain không thay thế cảm biến, vô lý - Sai.

3. Conclusion: Mặc dù bằng chứng hạn chế, đáp án hợp lý nhất dựa trên kiến thức công nghệ là A.

ANSWER: {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1}
"""

# Example 5: Comprehensive technical question
EXAMPLE_5 = """
EVIDENCE:
Transformer architecture sử dụng cơ chế self-attention để xử lý chuỗi dữ liệu. Không giống RNN xử lý tuần tự, Transformer xử lý song song toàn bộ chuỗi, cho phép huấn luyện nhanh hơn. Multi-head attention cho phép mô hình học nhiều representation khác nhau đồng thời.

QUESTION: Lợi ích chính của Transformer so với RNN là gì?

OPTIONS:
A. Sử dụng ít bộ nhớ hơn
B. Xử lý song song và huấn luyện nhanh hơn
C. Không cần dữ liệu huấn luyện nhiều
D. Luôn cho kết quả chính xác hơn

REASONING:
1. Evidence Analysis: Tài liệu nhấn mạnh Transformer xử lý song song toàn bộ chuỗi (khác với RNN xử lý tuần tự), dẫn đến huấn luyện nhanh hơn.

2. Option Evaluation:
   - Option A (Ít bộ nhớ hơn): Không được đề cập, thực tế Transformer thường tốn bộ nhớ hơn - Sai.
   - Option B (Xử lý song song và huấn luyện nhanh): Khớp chính xác với "xử lý song song toàn bộ chuỗi, cho phép huấn luyện nhanh hơn" - Đúng.
   - Option C (Không cần nhiều dữ liệu): Không được đề cập, thực tế Transformer cần nhiều dữ liệu - Sai.
   - Option D (Luôn chính xác hơn): Tuyệt đối hóa, không có bằng chứng - Sai.

3. Conclusion: Đáp án rõ ràng là B.

ANSWER: {"A": 0.0, "B": 1.0, "C": 0.0, "D": 0.0}
"""

# All examples in order
ALL_EXAMPLES = [EXAMPLE_1, EXAMPLE_2, EXAMPLE_3, EXAMPLE_4, EXAMPLE_5]

# Compact few-shot for small LLM (use 2 examples to save context)
COMPACT_EXAMPLES = [EXAMPLE_1, EXAMPLE_2]


def get_few_shot_examples(num_examples: int = 2, question_type: str = "general") -> str:
    """
    Get few-shot examples for CoT prompting.
    
    Args:
        num_examples: Number of examples to return (default 2 to save context)
        question_type: Type of question - 'general', 'multi', 'calculation', 'technical'
    
    Returns:
        Formatted few-shot examples string
    """
    if question_type == "multi":
        # Include multi-answer example
        examples = [EXAMPLE_1, EXAMPLE_3][:num_examples]
    elif question_type == "calculation":
        # Include calculation example
        examples = [EXAMPLE_2, EXAMPLE_1][:num_examples]
    elif question_type == "low_evidence":
        # Include low evidence example
        examples = [EXAMPLE_1, EXAMPLE_4][:num_examples]
    else:
        # General: use compact examples
        examples = COMPACT_EXAMPLES[:num_examples]
    
    return "\n\n".join(examples)


def detect_question_type(question: str) -> str:
    """
    Detect question type to select appropriate examples.
    
    Args:
        question: The question text
    
    Returns:
        Question type: 'multi', 'calculation', 'technical', or 'general'
    """
    question_lower = question.lower()
    
    # Check for multi-answer hints
    multi_hints = ["chọn tất cả", "chọn các", "select all", "which of the following"]
    if any(hint in question_lower for hint in multi_hints):
        return "multi"
    
    # Check for calculation indicators
    calc_hints = ["tính", "bao nhiêu", "calculate", "λ", "±", "=", "công thức"]
    if any(hint in question_lower for hint in calc_hints):
        return "calculation"
    
    return "general"
