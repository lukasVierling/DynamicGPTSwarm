#!/usr/bin/env python
# -*- coding: utf-8 -*-
# modified based on https://github.com/princeton-nlp/tree-of-thought-llm/blob/ab400345c5ea39d28ea6d7d3be0e417b11113c87/scripts/crosswords/search_crosswords-dfs.ipynb
import re
import json
'''
def parse_response(response):
    def parse_line(input_str):
        # regular expression pattern to match the input string format
        pattern = r'^([hv][1-5])\. ([a-zA-Z]{5,5}) \((certain|high|medium|low)\).*$'

        # use regex to extract the parts of the input string
        match = re.match(pattern, input_str)

        if match:
            # extract the matched groups
            parts = [match.group(1), match.group(2), match.group(3)]
            return parts
        else:
            return None
    # split the response into lines
    lines = response.split('\n')

    # parse each line
    parsed_lines = [parse_line(line) for line in lines]

    # filter out the lines that didn't match the format
    confidence_to_value = {'certain': 1, 'high': 0.5, 'medium': 0.2, 'low': 0.1} 
    parsed_lines = [(line[0].lower() + '. ' + line[1].lower(), confidence_to_value.get(line[2], 0)) for line in parsed_lines if line is not None]
    return sorted(parsed_lines, key=lambda x: x[1], reverse=True)
'''
def parse_response_JSON(response):
    pass


def parse_response(response):
    def parse_line(word, confidence):
        return word.lower(), confidence_to_value.get(confidence.lower(), 0)
    
    confidence_to_value = {'certain': 1, 'high': 0.5, 'medium': 0.2, 'low': 0.1} 

    # Convert the response JSON string to a Python dictionary
    #find first and last curly bracket
    first = response.find('{')
    last = response.rfind('}')
    response = response[first:last+1]
    try:
        response_dict = json.loads(response)
    except json.JSONDecodeError:
        print("Error: Unable to parse the response JSON string.")
        return []

    # Parse the "Filled" section
    filled_words = response_dict.get("Filled", {})
    parsed_lines = [(word.lower(), 1) for word in filled_words.values()]
    #filter all keys that are not of the shape: [hv][1-5] in the response dit
    response_dict = {k: v for k, v in response_dict.items() if re.match(r'^[hv][1-5]$', k)}
    # filter all values that are not of type List[List[str]] in the response dict
    response_dict = {k: v for k, v in response_dict.items() if isinstance(v, list) and all(isinstance(i, list) for i in v)}
    # filter all answers that are not of the shape [a-zA-Z]{5,5} in the response dict
    response_dict = {k: [w for w in v if re.match(r'^[a-zA-Z]{5,5}$', w[0])] for k, v in response_dict.items()}

    # Parse the word-confidence pairs for each position
    for position, word_confidence_pairs in response_dict.items():
        if position in ("Filled", "Changed"):
            continue
        
        for word, confidence in word_confidence_pairs:
            parsed_line = parse_line(word, confidence)
            parsed_lines.append((position.lower() + '. ' + parsed_line[0], parsed_line[1]))

    # Sort the parsed lines based on confidence in descending order
    parsed_lines.sort(key=lambda x: x[1], reverse=True)

    return parsed_lines