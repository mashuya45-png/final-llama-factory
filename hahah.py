def evaluate_response_quality(response, question, category):
    """综合评估回答质量，包含四个维度评分"""
    # 1. 相关性评分
    relevance_score = min(len(response) / 100, 1.0)
    # 2. 关键词匹配评分
    keyword_score = calculate_keyword_match(response, category)
    # 3. 结构完整性评分
    structure_score = evaluate_structure(response)
    # 4. 信息密度评分
    info_score = calculate_information_density(response)
    # 综合评分
    total_score = (relevance_score * 0.3 + keyword_score * 0.3 +
                   structure_score * 0.2 + info_score * 0.2)
    return total_score