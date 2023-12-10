class BasicTokenizer(object):
  """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

  def __init__(self, do_lower_case=True):
    """Constructs a BasicTokenizer.

    Args:
      do_lower_case: 是否将 query 字母都转化为小写 
    """
    self.do_lower_case = do_lower_case

  def tokenize(self, text):
    """Tokenizes a piece of text."""
    # step 1：将 text 从 Unicode 转化为 utf-8
    text = convert_to_unicode(text)
    # step 2：去除无意义字符以及空格
    text = self._clean_text(text)
    # step 3：增加中文支持
    text = self._tokenize_chinese_chars(text)
    # step 4：在一段文本上运行基本的空格清除和拆分
    orig_tokens = whitespace_tokenize(text)
    # step 5：用标点切分
    split_tokens = []
    for token in orig_tokens:
      # 是否转小写
      if self.do_lower_case:
        token = token.lower()
        # 对text进行归一化
        token = self._run_strip_accents(token)
      # 用标点切分
      split_tokens.extend(self._run_split_on_punc(token))
    # step 5：在一段文本上运行基本的空格清除和拆分
    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return output_tokens

  def _run_strip_accents(self, text):
    """这个函数去除掉text中的非间距字符"""
    # step 1： 对text进行归一化
    # 标准化对于任何需要以一致的方式处理Unicode文本的程序都是非常重要的。
    # 当处理来自用户输入的字符串而你很难去控制编码的时候尤其如此。
    # normalize() 将文本标准化,第一个参数指定字符串标准化的方式,NFD表示字符应该分解为多个组合字符表示
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      # # category() 返回字符在UNICODE里分类的类型
      cat = unicodedata.category(char)
      # 判断cat 是否为 Mn，Mark, Nonspacing 指示字符是非间距字符，这指示基字符的修改。
      if cat == "Mn":
        continue
      output.append(char)
    return "".join(output)

  def _run_split_on_punc(self, text):
    """用标点对 文本 进行切分，返回list"""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ["".join(x) for x in output]

  def _tokenize_chinese_chars(self, text):
    """ 按字切分中文，实现就是在字两侧添加空格 
    Adds whitespace around any CJK character. """
    output = []
    for char in text:
      cp = ord(char)
      if self._is_chinese_char(cp):
        output.append(" ")
        output.append(char)
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

  def _is_chinese_char(self, cp):
    """ 判断是否是汉字
    Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False

  def _clean_text(self, text):
    """
      去除无意义字符以及空格
      Performs invalid character removal and whitespace cleanup on text.
    """
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

