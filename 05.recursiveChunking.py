import re

def recursive_chunk(text, max_size, level=0):
    '''
    Recursive chunk the text into smaller parts using a set of separators.
    
    Parameters:
    text(str) : The input text to be chunked.
    max_size(int): The maximum desired chunk size.
    level(int): The current recursion level(used for debugging purpose).
    
    Returns:
    list: A list of text chunks.
    '''
    
    # Define seperators for different levels of chunking
    seperators = [r'(?<=[.!?])+',r'\s+'] # Sentence level, word level
    
    # If the text is already within the max size, return it as a single chunk
    if len(text) <= max_size:
        return [text]
    
    # Select the appropriate separator based on the recursion level
    seperator = seperators[min(level, len(seperators) - 1)]
    
    # Split the text using the selected separator
    chunks = re.split(seperator, text)
    
    # If the number of chunks is too large, recursively split each chunk
    if any(len(chunk) > max_size for chunk in chunks):
        new_chunks = []
        for chunk in chunks:
            if len(chunk) > max_size:
                new_chunks.extend(recursive_chunk(chunk, max_size, level + 1))
            else:
                new_chunks.append(chunk)
                
        return new_chunks
    else:
        return chunks
    
# Sample text
text = (
    "Recursive chunking divides the text hierarchically using a set of separators."
    "If the initial chunks are too large, the method recursively splits them until "
    "the desired size is achived. This technique is useful for processing large "
    "texts where simpler chunking methods may fail. Let's see how it works."
)

# Desired maximum chunk size (number of characters)
max_size = 50

# Recursively chunk the text
chunks = recursive_chunk(text, max_size)

# Print the chunks
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}: \n{chunk}\n")