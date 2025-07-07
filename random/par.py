
#par
import os
import numpy as np


# Define the directory where the files are located
directory = '.'

# Generate the list of file names based on the pattern
file_names1 = [f'1.{i}.0.xml.out' for i in range(240, 481, 5)]
file_names2 = [f'1.{i}.0.xml.out' for i in range(242, 478, 5)]
# Combine the lists
combined_file_names = file_names1 + file_names2

# Remove duplicates (optional)
combined_file_names = list(set(combined_file_names))

# Sort by the number after the first decimal point
sorted_file_names = sorted(combined_file_names, key=lambda x: int(x.split('.')[1]))

# Combine the lists and remove duplicates, sorting by the number after the first decimal
combined_file_names = sorted(set(file_names1 + file_names2), key=lambda x: int(x.split('.')[1]))

# Initialize an empty list to store sigma_par columns
sigma_par_matrix = []

# Loop through each file name
for combined_file_name in combined_file_names:
    sigma_par_values = []

    # Construct the full path to the file
    file_path = os.path.join(directory, combined_file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    try:
        # Open the file and read its contents
        with open(file_path, 'r') as file:
            # Search for the header containing "Sigma_par"
            for line in file:
                if 'Sigma_par' in line:
                    # Identify the column index for 'Sigma_par'
                    header_columns = line.split()
                    sigma_par_index = header_columns.index('Sigma_par')
                    break
            else:
                print(f"Expected header not found in file: {file_path}")
                continue

            # Read the rest of the file line by line
            for line in file:
                # Stop reading if a termination message is encountered
                if line.startswith("Job complete.") or line.startswith("Job time:"):
                    break

                # Split line into columns
                values = line.split()
                
                # Ensure line has enough columns to avoid index error
                if len(values) > sigma_par_index:
                    sigma_par = values[sigma_par_index]
                    try:
                        sigma_par_values.append(float(sigma_par))  # Convert to float
                    except ValueError as e:
                        print(f"Error converting '{sigma_par}' to float in file: {file_path}. Error: {e}")

    except IOError as e:
        print(f"Error reading file {file_path}: {e}")

    # Append the column of sigma_par values to the matrix if not empty
    if sigma_par_values:
        sigma_par_matrix.append(sigma_par_values)
    else:
        print(f"No valid Sigma_par values found in file: {file_path}")

par0 = np.array(sigma_par_matrix).T


# Generate the list of file names based on the pattern
file_names1 = [f'1.{i}.1.xml.out' for i in range(240, 481, 5)]
file_names2 = [f'1.{i}.1.xml.out' for i in range(242, 478, 5)]
# Combine the lists
combined_file_names = file_names1 + file_names2
# Remove duplicates (optional)
combined_file_names = list(set(combined_file_names))

# Sort by the number after the first decimal point
sorted_file_names = sorted(combined_file_names, key=lambda x: int(x.split('.')[1]))
# Combine the lists and remove duplicates, sorting by the number after the first decimal
combined_file_names = sorted(set(file_names1 + file_names2), key=lambda x: int(x.split('.')[1]))

# Initialize an empty list to store sigma_par columns
sigma_par_matrix = []

# Loop through each file name
for combined_file_name in combined_file_names:
    sigma_par_values = []

    # Construct the full path to the file
    file_path = os.path.join(directory, combined_file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    try:
        # Open the file and read its contents
        with open(file_path, 'r') as file:
            # Search for the header containing "Sigma_par"
            for line in file:
                if 'Sigma_par' in line:
                    # Identify the column index for 'Sigma_par'
                    header_columns = line.split()
                    sigma_par_index = header_columns.index('Sigma_par')
                    break
            else:
                print(f"Expected header not found in file: {file_path}")
                continue

            # Read the rest of the file line by line
            for line in file:
                # Stop reading if a termination message is encountered
                if line.startswith("Job complete.") or line.startswith("Job time:"):
                    break

                # Split line into columns
                values = line.split()
                
                # Ensure line has enough columns to avoid index error
                if len(values) > sigma_par_index:
                    sigma_par = values[sigma_par_index]
                    try:
                        sigma_par_values.append(float(sigma_par))  # Convert to float
                    except ValueError as e:
                        print(f"Error converting '{sigma_par}' to float in file: {file_path}. Error: {e}")

    except IOError as e:
        print(f"Error reading file {file_path}: {e}")

    # Append the column of sigma_par values to the matrix if not empty
    if sigma_par_values:
        sigma_par_matrix.append(sigma_par_values)
    else:
        print(f"No valid Sigma_par values found in file: {file_path}")
        
par1 = np.array(sigma_par_matrix).T

# Generate the list of file names based on the pattern
file_names1 = [f'1.{i}.2.xml.out' for i in range(240, 481, 5)]
file_names2 = [f'1.{i}.2.xml.out' for i in range(242, 478, 5)]
# Combine the lists
combined_file_names = file_names1 + file_names2

# Remove duplicates (optional)
combined_file_names = list(set(combined_file_names))

# Sort by the number after the first decimal point
sorted_file_names = sorted(combined_file_names, key=lambda x: int(x.split('.')[1]))

# Combine the lists and remove duplicates, sorting by the number after the first decimal
combined_file_names = sorted(set(file_names1 + file_names2), key=lambda x: int(x.split('.')[1]))

# Initialize an empty list to store sigma_par columns
sigma_par_matrix = []

# Loop through each file name
for combined_file_name in combined_file_names:
    sigma_par_values = []

    # Construct the full path to the file
    file_path = os.path.join(directory, combined_file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    try:
        # Open the file and read its contents
        with open(file_path, 'r') as file:
            # Search for the header containing "Sigma_par"
            for line in file:
                if 'Sigma_par' in line:
                    # Identify the column index for 'Sigma_par'
                    header_columns = line.split()
                    sigma_par_index = header_columns.index('Sigma_par')
                    break
            else:
                print(f"Expected header not found in file: {file_path}")
                continue

            # Read the rest of the file line by line
            for line in file:
                # Stop reading if a termination message is encountered
                if line.startswith("Job complete.") or line.startswith("Job time:"):
                    break

                # Split line into columns
                values = line.split()
                
                # Ensure line has enough columns to avoid index error
                if len(values) > sigma_par_index:
                    sigma_par = values[sigma_par_index]
                    try:
                        sigma_par_values.append(float(sigma_par))  # Convert to float
                    except ValueError as e:
                        print(f"Error converting '{sigma_par}' to float in file: {file_path}. Error: {e}")

    except IOError as e:
        print(f"Error reading file {file_path}: {e}")

    # Append the column of sigma_par values to the matrix if not empty
    if sigma_par_values:
        sigma_par_matrix.append(sigma_par_values)
    else:
        print(f"No valid Sigma_par values found in file: {file_path}")

par2 = np.array(sigma_par_matrix).T

# Generate the list of file names based on the pattern
file_names1 = [f'1.{i}.3.xml.out' for i in range(240, 481, 5)]
file_names2 = [f'1.{i}.3.xml.out' for i in range(242, 478, 5)]
# Combine the lists
combined_file_names = file_names1 + file_names2
# Combine the lists

# Remove duplicates (optional)
combined_file_names = list(set(combined_file_names))

# Sort by the number after the first decimal point
sorted_file_names = sorted(combined_file_names, key=lambda x: int(x.split('.')[1]))
# Combine the lists and remove duplicates, sorting by the number after the first decimal
combined_file_names = sorted(set(file_names1 + file_names2), key=lambda x: int(x.split('.')[1]))

# Initialize an empty list to store sigma_par columns
sigma_par_matrix = []

# Loop through each file name
for combined_file_name in combined_file_names:
    sigma_par_values = []

    # Construct the full path to the file
    file_path = os.path.join(directory, combined_file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    try:
        # Open the file and read its contents
        with open(file_path, 'r') as file:
            # Search for the header containing "Sigma_par"
            for line in file:
                if 'Sigma_par' in line:
                    # Identify the column index for 'Sigma_par'
                    header_columns = line.split()
                    sigma_par_index = header_columns.index('Sigma_par')
                    break
            else:
                print(f"Expected header not found in file: {file_path}")
                continue

            # Read the rest of the file line by line
            for line in file:
                # Stop reading if a termination message is encountered
                if line.startswith("Job complete.") or line.startswith("Job time:"):
                    break

                # Split line into columns
                values = line.split()
                
                # Ensure line has enough columns to avoid index error
                if len(values) > sigma_par_index:
                    sigma_par = values[sigma_par_index]
                    try:
                        sigma_par_values.append(float(sigma_par))  # Convert to float
                    except ValueError as e:
                        print(f"Error converting '{sigma_par}' to float in file: {file_path}. Error: {e}")

    except IOError as e:
        print(f"Error reading file {file_path}: {e}")

    # Append the column of sigma_par values to the matrix if not empty
    if sigma_par_values:
        sigma_par_matrix.append(sigma_par_values)
    else:
        print(f"No valid Sigma_par values found in file: {file_path}")
        
par3 = np.array(sigma_par_matrix).T



# Generate the list of file names based on the pattern
file_names1 = [f'1.{i}.4.xml.out' for i in range(240, 481, 5)]
file_names2 = [f'1.{i}.4.xml.out' for i in range(242, 478, 5)]
# Combine the lists
combined_file_names = file_names1 + file_names2

# Remove duplicates (optional)
combined_file_names = list(set(combined_file_names))

# Sort by the number after the first decimal point
sorted_file_names = sorted(combined_file_names, key=lambda x: int(x.split('.')[1]))

# Combine the lists and remove duplicates, sorting by the number after the first decimal
combined_file_names = sorted(set(file_names1 + file_names2), key=lambda x: int(x.split('.')[1]))

# Initialize an empty list to store sigma_par columns
sigma_par_matrix = []

# Loop through each file name
for combined_file_name in combined_file_names:
    sigma_par_values = []

    # Construct the full path to the file
    file_path = os.path.join(directory, combined_file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    try:
        # Open the file and read its contents
        with open(file_path, 'r') as file:
            # Search for the header containing "Sigma_par"
            for line in file:
                if 'Sigma_par' in line:
                    # Identify the column index for 'Sigma_par'
                    header_columns = line.split()
                    sigma_par_index = header_columns.index('Sigma_par')
                    break
            else:
                print(f"Expected header not found in file: {file_path}")
                continue

            # Read the rest of the file line by line
            for line in file:
                # Stop reading if a termination message is encountered
                if line.startswith("Job complete.") or line.startswith("Job time:"):
                    break

                # Split line into columns
                values = line.split()
                
                # Ensure line has enough columns to avoid index error
                if len(values) > sigma_par_index:
                    sigma_par = values[sigma_par_index]
                    try:
                        sigma_par_values.append(float(sigma_par))  # Convert to float
                    except ValueError as e:
                        print(f"Error converting '{sigma_par}' to float in file: {file_path}. Error: {e}")

    except IOError as e:
        print(f"Error reading file {file_path}: {e}")

    # Append the column of sigma_par values to the matrix if not empty
    if sigma_par_values:
        sigma_par_matrix.append(sigma_par_values)
    else:
        print(f"No valid Sigma_par values found in file: {file_path}")

par4 = np.array(sigma_par_matrix).T


# Generate the list of file names based on the pattern
file_names1 = [f'1.{i}.5.xml.out' for i in range(240, 481, 5)]
file_names2 = [f'1.{i}.5.xml.out' for i in range(242, 478, 5)]
# Combine the lists
combined_file_names = file_names1 + file_names2

# Remove duplicates (optional)
combined_file_names = list(set(combined_file_names))

# Sort by the number after the first decimal point
sorted_file_names = sorted(combined_file_names, key=lambda x: int(x.split('.')[1]))

# Combine the lists and remove duplicates, sorting by the number after the first decimal
combined_file_names = sorted(set(file_names1 + file_names2), key=lambda x: int(x.split('.')[1]))

# Initialize an empty list to store sigma_par columns
sigma_par_matrix = []

# Loop through each file name
for combined_file_name in combined_file_names:
    sigma_par_values = []

    # Construct the full path to the file
    file_path = os.path.join(directory, combined_file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    try:
        # Open the file and read its contents
        with open(file_path, 'r') as file:
            # Search for the header containing "Sigma_par"
            for line in file:
                if 'Sigma_par' in line:
                    # Identify the column index for 'Sigma_par'
                    header_columns = line.split()
                    sigma_par_index = header_columns.index('Sigma_par')
                    break
            else:
                print(f"Expected header not found in file: {file_path}")
                continue

            # Read the rest of the file line by line
            for line in file:
                # Stop reading if a termination message is encountered
                if line.startswith("Job complete.") or line.startswith("Job time:"):
                    break

                # Split line into columns
                values = line.split()
                
                # Ensure line has enough columns to avoid index error
                if len(values) > sigma_par_index:
                    sigma_par = values[sigma_par_index]
                    try:
                        sigma_par_values.append(float(sigma_par))  # Convert to float
                    except ValueError as e:
                        print(f"Error converting '{sigma_par}' to float in file: {file_path}. Error: {e}")

    except IOError as e:
        print(f"Error reading file {file_path}: {e}")

    # Append the column of sigma_par values to the matrix if not empty
    if sigma_par_values:
        sigma_par_matrix.append(sigma_par_values)
    else:
        print(f"No valid Sigma_par values found in file: {file_path}")

par5 = np.array(sigma_par_matrix).T

