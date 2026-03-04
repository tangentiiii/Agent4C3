### persona_generator.py

+ `format_user_history`中，将过长的post/comment做了截断，post保留300字符，comments保留200字符

+ `parse_persona`中，从LLM生成的persona中提取信息。提取信息的方式：

  ```python
  style_match = re.search(r"CONVERSATIONAL_STYLE:\s*(.+)", raw_text)
      if style_match:
          persona["conversational_style"] = style_match.group(1).strip()
  ```

  是一种按行提取的方式，如果LLM不是按照这种格式输出可能会出问题



### synthesize_click_like.yml/synthetic_data.py

1.只使用了posts(而没有使用comments)来合成点赞数据

2.<同一个user多个posts>放在了同一个prompt/同一次LLM_request当中，不知道是否可行

3.click & like 在同一个prompt中，是否需要分开？

4.LLM输出格式感觉不是很清楚



输出的结果`click_like_data.json`中，出现以下问题：

+ `"text": "1."`, text的值只是一个序号
+ `"text": "[deleted]"`, text的值是一个[deleted] tag
+ `"text": ""`,text的值是空串
+ `"text": "[removed]"`, text的值是一个[removed] tag

需要添加数据的过滤。



### user_click.yml

```yaml
Respond with a JSON object: {{"clicks": [<list of 0-indexed post numbers you would click>]}}
```

LLM能够按照要求输出吗？

没有要求输出reason，需不需要？



