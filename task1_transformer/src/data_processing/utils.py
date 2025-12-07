import re

def clean_text(text: str) -> str:
    """
    Chuẩn hóa văn bản: chuyển về chữ thường, xóa ký tự đặc biệt và xử lý khoảng trắng thừa.
    
    Args:
        text (str): Chuỗi văn bản đầu vào.
        
    Returns:
        str: Chuỗi văn bản đã được làm sạch.
    """
    text = str(text).lower()
    text = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", text)
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r"\!+", "!", text)
    text = re.sub(r"\,+", ",", text)
    text = re.sub(r"\?+", "?", text)
    return text.strip()

if __name__ == "__main__":
    sample = "HEllo*** world!!!   How are you?"
    print(f"Original: {sample}")
    print(f"Cleaned:  {clean_text(sample)}")