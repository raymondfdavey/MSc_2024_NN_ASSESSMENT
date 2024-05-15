###
### To use this word counting tool, issue this command in the command line:
### 
### python3 count_jupyter_nb_words.py XXXXXX.ipynb
### 
### where XXXXXX.ipynb is the name of your Notebook file, and XXXXXX is your candidate number.
### 


# The updates include these things:

# Please check the file "useful_code_for_figures_equations.ipynb". This explains some subtle changes to how you can generate figures and Latex formatted equations that will hopefully not contribute to the word count. The word counting script looks for tags, either <lt> for latex or <figure> for figures, and excludes the related markdown text. Previously, I had suggested formatting some Latex like this: 

# $$
# some latex code
# $$
# or formatting markdown code for figures like this:
# <figure>
# some figure details
# </figure>

# Now, all markdown to produce a single figure must be on the same line if the word counter is to ignore it, i.e. do not use any carriage returns between $ and $, or $$ and $$ for latex, or between <figure> and </figure> for figures.  E.g. figures must be formatted as such
# <figure> some figure details </figure>
# and Latex must be formatted as such:
# $.....$
# $$..........$$ 
# The word counter no longer includes cells with instructions, and will only count from cells where you enter text. Because this requires special tags to ignore certain cells, you will need to use the new XXXXXX.ipynb file, not the old one. If you have already done a substantial amount of work that would take a long time to rerun in the new file, you may enter this tag:
# <ignore>
# in the cells that include my instructions. !!!BUT DO SO CAREFULLY!!! Some of those cells include instructions in addtition to a prompt for you to enter your own text. You will have to create a new cell to separate your text from the instructions. And just in case you are tempted to use this tag to give the impression of a lower word count (of course you wouldn't!), I know exactly how many <ignore> tags there should be if you need to add them to the old XXXXXX.ipynb file.
# I have updated last section of the assignment brief, specifically the description for the Marking Criteria under the 70%-100% range. Before, it was stated that you needed to incorporate additional features to you model, or to do more than what was being asked of you, in order to score marks in this range. This was my mistake, and was based on last year's assignment, which took a slightly different format. I have now removed those specific statements from the marking criteria as they are not relevant to this year's assessment. The description that remains holds true, however.



import nbformat
import sys
import re

def count_words_in_markdown(notebook_file):
    # Load the notebook
    with open(notebook_file, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)

    # Iterate through each cell and count words in markdown cells
    limit = 2500
    words_to_be_replaced = 80
    word_count = 0

    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            # Extract text from the markdown cell
            cell_content = cell.source
            if '<ignore>' in cell_content:
                continue
            else:
                # Remove frmatting that should not contribute towards the word count.
                cell_content = re.sub('\<(lt)\>\$[^\$]+\$','',cell_content) # Remove in-line Latex (flagged by <lt>$.$)
                cell_content = re.sub('\<(lt)\>(\$){2}[^\$]+(\$){2}','',cell_content) # Remove newline Latex (flagged by <lt>$$.$$)
                cell_content = re.sub('\<(figure)\>.*\<(\/figure)\>','',cell_content) # Remove figures (flagged by <figure>.</figure>)
                # Count words remaining in this cell
                cell_count = len(cell_content.split())
                # Update word count
                word_count += cell_count
                # Check whether word count exceeds limit
                if (word_count > limit) and ((word_count - cell_count)<limit):
                    print("stop marking from here: {}" .format(' '.join(cell['source'].split()[(-(word_count - limit)-5):(-(word_count - limit))])))
            
    return word_count

fname = sys.argv[1] # Provide notebook file name
word_count = count_words_in_markdown(fname)
print(f"Total number of words in markdown cells (ignoring LaTeX and figures): {word_count}")
