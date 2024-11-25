#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from Bio.SeqUtils import gc_fraction 
import numpy as np
import subprocess
import re


# In[2]:


fp = "/home/myoui/shared/ML/Homo_sapiens.GRCh38.ncrna.fa"


# In[3]:


data_lists = [[] for _ in range(8)]  # Creates a total of 8 lists for columns one to eight
sequence_list = []
with open(fp, 'r') as file:
    current_sequence = []  
    for line in file:
        line = line.strip()  # Removes whitespace

        # Processes header lines starting with '>'
        if line.startswith('>'):
            # Before processing the new header, appends the current sequence if it exists
            if current_sequence:
                sequence_list.append(''.join(current_sequence))  # Joins and appends sequence
                current_sequence = []  # Resets for the next sequence

            # Splits line into fields
            parts = line.split()

            # Appends to respective lists only if there are at least 8 parts
            for i in range(8):
                if i < len(parts):
                    data_lists[i].append(parts[i].replace('>', '') if i == 0 else parts[i])
                else:
                    data_lists[i].append(None)  # Fills with None if the part doesn't exist

        else:
            # If it's a sequence line (not a header), accumulate the sequence
            current_sequence.append(line)

    # Append the last sequence after the loop
    if current_sequence:
        sequence_list.append(''.join(current_sequence))

# Prints lengths to check for mismatches
for idx, lst in enumerate(data_lists):
    print(f"Column {idx + 1} list length: {len(lst)}")
print(f"Sequence list length: {len(sequence_list)}")


# In[4]:


# Save the parsed data in dataframe
df = pd.DataFrame(data_lists).T


# In[5]:


# Columns for Gene list dataframe
transcript_id = df[0]
chromosome_location = df[2]
ensembl_id = df[3]
gene_type = df[4]
gene_symbol = df[6]

# Removes prefixes
ensembl_id = ensembl_id.str.replace('gene:', '', regex=False)
gene_type = gene_type.str.replace('gene_biotype:', '', regex=False)
gene_symbol = gene_symbol.str.replace('gene_symbol:', '', regex=False)


# In[6]:


# Create Gene list dataframe
gene_list = pd.DataFrame({
    'transcript_id' : transcript_id,
    'ensembl_id': ensembl_id,
    'chromosome_location' : chromosome_location,
    'gene_type' : gene_type,
    'gene_symbol' : gene_symbol,
    'sequence' : sequence_list
})


# In[7]:


# Save the DataFrame to CSV
output = '/home/myoui/shared/ML/Gene_list.csv'
gene_list.to_csv(output, index=False)


# In[8]:


gene_list['gene_type'].unique()


# In[9]:


gene_list['gene_type'].value_counts()


# In[10]:


# Categories with significant no. of genes
categories = ['lncRNA', 'misc_RNA', 'snRNA', 'miRNA', 'snoRNA']


# In[11]:


# Filter only gene categories except lncRNA
gene_filtered = gene_list[gene_list['gene_type'].isin(categories) & (gene_list['gene_type'] != 'lncRNA')]


# In[12]:


# Randomly select 2000 genes with random_state = 42 for reproducibility
lncrna_sample = gene_list[gene_list['gene_type'] == 'lncRNA'].sample(n=2000, random_state=42)


# In[13]:


# Add lncRNA
gene_filtered = pd.concat([gene_filtered, lncrna_sample])


# In[14]:


# Get GC content and Sequence length
def gc_content(seq):
    return gc_fraction(seq)

def sequence_length(seq):
    return len(seq)

gene_filtered['gc_content'] = gene_filtered['sequence'].apply(gc_content)
gene_filtered['seq_length'] = gene_filtered['sequence'].apply(sequence_length)


# In[15]:


# Extract genes above 4000 nucleotides
large_genes = gene_filtered[gene_filtered['seq_length'] > 4000]
large_genes


# In[16]:


# Remove ENST00000717492.1, ENST00000650366.1 and ENST00000643616.1 from dataframe(large, causes issues with RNAfold)
transcripts_to_remove = ['ENST00000717492.1', 'ENST00000650366.1', 'ENST00000643616.1']
gene_filtered = gene_filtered[~gene_filtered['transcript_id'].isin(transcripts_to_remove)]


# In[17]:


# Initialize dictionaries to store structure and mfe
structure_dict = {}
mfe_dict = {}

# Open and parse the RNAfold output file
with open('/home/myoui/shared/ML/rnafold_output.txt', 'r') as f:
    lines = f.readlines()
    
    for i in range(0, len(lines), 3):  # Read the file in chunks of 3 lines
        # Extract transcript ID (remove '>' from the beginning)
        transcript_id = lines[i].strip().replace('>', '')
        
        # Extract structure and MFE from line i+2
        structure_line = lines[i+2].strip()
        structure = structure_line.split(" ")[0]
        
        # Check if MFE is present and can be converted to float
        try:
            mfe = float(structure_line.split(" ")[1].strip("()"))
        except (IndexError, ValueError):
            mfe = None  # Assign None if MFE is missing or can't be converted
        
        # Store structure and mfe in dictionaries
        structure_dict[transcript_id] = structure
        mfe_dict[transcript_id] = mfe

# Add structure and mfe to gene_filtered dataframe by matching transcript_id
gene_filtered['structure'] = gene_filtered['transcript_id'].map(structure_dict)
gene_filtered['mfe'] = gene_filtered['transcript_id'].map(mfe_dict)


# In[18]:


gene_filtered['dot_count'] = gene_filtered['structure'].str.count(r'\.')
gene_filtered['bracket_count'] = gene_filtered['structure'].str.count(r'\(') + gene_filtered['structure'].str.count(r'\)')


# In[19]:


data = gene_filtered.dropna(subset=['mfe'])


# In[20]:


def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

# Generate k-mers and add them as a feature
data['kmers'] = data['sequence'].apply(lambda x: getKmers(x))

# Combine k-mers into a single space-separated string per sequence
data['kmers'] = data['kmers'].apply(lambda x: ' '.join(x))


# In[21]:


data


# In[22]:


data.to_csv('/home/myoui/shared/ML/gene_list_para.csv', index=False)


# In[23]:


df = gene_list
gene_types_of_interest = ['scRNA', 'Mt_rRNA', 'vault_RNA']
filtered_df = df[df['gene_type'].isin(gene_types_of_interest)]
print(filtered_df)
filtered_df.to_csv('/home/myoui/shared/ML/scRNA_Mt_vault.csv', index=False)


# In[ ]:





# In[ ]:




