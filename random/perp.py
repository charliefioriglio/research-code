
#perp
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

# Initialize an empty list to store sigma_perp columns
sigma_perp_matrix = []

# Loop through each file name
for combined_file_name in combined_file_names:
    sigma_perp_values = []

    # Construct the full path to the file
    file_path = os.path.join(directory, combined_file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    try:
        # Open the file and read its contents
        with open(file_path, 'r') as file:
            # Search for the header containing "Sigma_perp"
            for line in file:
                if 'Sigma_perp' in line:
                    # Identify the column index for 'Sigma_perp'
                    header_columns = line.split()
                    sigma_perp_index = header_columns.index('Sigma_perp')
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
                if len(values) > sigma_perp_index:
                    sigma_perp = values[sigma_perp_index]
                    try:
                        sigma_perp_values.append(float(sigma_perp))  # Convert to float
                    except ValueError as e:
                        print(f"Error converting '{sigma_perp}' to float in file: {file_path}. Error: {e}")

    except IOError as e:
        print(f"Error reading file {file_path}: {e}")

    # Append the column of sigma_perp values to the matrix if not empty
    if sigma_perp_values:
        sigma_perp_matrix.append(sigma_perp_values)
    else:
        print(f"No valid Sigma_perp values found in file: {file_path}")

perp0 = np.array(sigma_perp_matrix).T


# Generate the list of file names based on the pattern

# Generate the list of file names based on the pattern
file_names1 = [f'1.{i}.1.xml.out' for i in range(240, 481, 5)]
file_names2 = [f'1.{i}.1.xml.out' for i in range(242, 478, 5)]
# Combine the lists
combined_file_names = file_names1 + file_names2

# Remove duplicates (optional)
combined_file_names = list(set(combined_file_names))

# Sort by the number after the first decimal point
sorted_file_names = sorted(combined_file_names, key=lambda x: int(x.split('.')[1]))

# Initialize an empty list to store sigma_perp columns
sigma_perp_matrix = []

# Loop through each file name
for combined_file_name in combined_file_names:
    sigma_perp_values = []

    # Construct the full path to the file
    file_path = os.path.join(directory, combined_file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    try:
        # Open the file and read its contents
        with open(file_path, 'r') as file:
            # Search for the header containing "Sigma_perp"
            for line in file:
                if 'Sigma_perp' in line:
                    # Identify the column index for 'Sigma_perp'
                    header_columns = line.split()
                    sigma_perp_index = header_columns.index('Sigma_perp')
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
                if len(values) > sigma_perp_index:
                    sigma_perp = values[sigma_perp_index]
                    try:
                        sigma_perp_values.append(float(sigma_perp))  # Convert to float
                    except ValueError as e:
                        print(f"Error converting '{sigma_perp}' to float in file: {file_path}. Error: {e}")

    except IOError as e:
        print(f"Error reading file {file_path}: {e}")

    # Append the column of sigma_perp values to the matrix if not empty
    if sigma_perp_values:
        sigma_perp_matrix.append(sigma_perp_values)
    else:
        print(f"No valid Sigma_perp values found in file: {file_path}")

perp1 = np.array(sigma_perp_matrix).T



# Generate the list of file names based on the pattern

# Generate the list of file names based on the pattern
file_names1 = [f'1.{i}.2.xml.out' for i in range(240, 481, 5)]
file_names2 = [f'1.{i}.2.xml.out' for i in range(242, 478, 5)]
# Combine the lists
combined_file_names = file_names1 + file_names2

# Remove duplicates (optional)
combined_file_names = list(set(combined_file_names))

# Sort by the number after the first decimal point
sorted_file_names = sorted(combined_file_names, key=lambda x: int(x.split('.')[1]))

# Initialize an empty list to store sigma_perp columns
sigma_perp_matrix = []

# Loop through each file name
for combined_file_name in combined_file_names:
    sigma_perp_values = []

    # Construct the full path to the file
    file_path = os.path.join(directory, combined_file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    try:
        # Open the file and read its contents
        with open(file_path, 'r') as file:
            # Search for the header containing "Sigma_perp"
            for line in file:
                if 'Sigma_perp' in line:
                    # Identify the column index for 'Sigma_perp'
                    header_columns = line.split()
                    sigma_perp_index = header_columns.index('Sigma_perp')
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
                if len(values) > sigma_perp_index:
                    sigma_perp = values[sigma_perp_index]
                    try:
                        sigma_perp_values.append(float(sigma_perp))  # Convert to float
                    except ValueError as e:
                        print(f"Error converting '{sigma_perp}' to float in file: {file_path}. Error: {e}")

    except IOError as e:
        print(f"Error reading file {file_path}: {e}")

    # Append the column of sigma_perp values to the matrix if not empty
    if sigma_perp_values:
        sigma_perp_matrix.append(sigma_perp_values)
    else:
        print(f"No valid Sigma_perp values found in file: {file_path}")
perp2 = np.array(sigma_perp_matrix).T


# Generate the list of file names based on the pattern
file_names1 = [f'1.{i}.3.xml.out' for i in range(240, 481, 5)]
file_names2 = [f'1.{i}.3.xml.out' for i in range(242, 478, 5)]
# Combine the lists
combined_file_names = file_names1 + file_names2

# Remove duplicates (optional)
combined_file_names = list(set(combined_file_names))

# Sort by the number after the first decimal point
sorted_file_names = sorted(combined_file_names, key=lambda x: int(x.split('.')[1]))

# Initialize an empty list to store sigma_perp columns
sigma_perp_matrix = []

# Loop through each file name
for combined_file_name in combined_file_names:
    sigma_perp_values = []

    # Construct the full path to the file
    file_path = os.path.join(directory, combined_file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    try:
        # Open the file and read its contents
        with open(file_path, 'r') as file:
            # Search for the header containing "Sigma_perp"
            for line in file:
                if 'Sigma_perp' in line:
                    # Identify the column index for 'Sigma_perp'
                    header_columns = line.split()
                    sigma_perp_index = header_columns.index('Sigma_perp')
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
                if len(values) > sigma_perp_index:
                    sigma_perp = values[sigma_perp_index]
                    try:
                        sigma_perp_values.append(float(sigma_perp))  # Convert to float
                    except ValueError as e:
                        print(f"Error converting '{sigma_perp}' to float in file: {file_path}. Error: {e}")

    except IOError as e:
        print(f"Error reading file {file_path}: {e}")

    # Append the column of sigma_perp values to the matrix if not empty
    if sigma_perp_values:
        sigma_perp_matrix.append(sigma_perp_values)
    else:
        print(f"No valid Sigma_perp values found in file: {file_path}")

perp3 = np.array(sigma_perp_matrix).T


# Generate the list of file names based on the pattern
file_names1 = [f'1.{i}.4.xml.out' for i in range(240, 481, 5)]
file_names2 = [f'1.{i}.4.xml.out' for i in range(242, 478, 5)]
# Combine the lists
combined_file_names = file_names1 + file_names2

# Remove duplicates (optional)
combined_file_names = list(set(combined_file_names))

# Sort by the number after the first decimal point
sorted_file_names = sorted(combined_file_names, key=lambda x: int(x.split('.')[1]))

# Initialize an empty list to store sigma_perp columns
sigma_perp_matrix = []

# Loop through each file name
for combined_file_name in combined_file_names:
    sigma_perp_values = []

    # Construct the full path to the file
    file_path = os.path.join(directory, combined_file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    try:
        # Open the file and read its contents
        with open(file_path, 'r') as file:
            # Search for the header containing "Sigma_perp"
            for line in file:
                if 'Sigma_perp' in line:
                    # Identify the column index for 'Sigma_perp'
                    header_columns = line.split()
                    sigma_perp_index = header_columns.index('Sigma_perp')
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
                if len(values) > sigma_perp_index:
                    sigma_perp = values[sigma_perp_index]
                    try:
                        sigma_perp_values.append(float(sigma_perp))  # Convert to float
                    except ValueError as e:
                        print(f"Error converting '{sigma_perp}' to float in file: {file_path}. Error: {e}")

    except IOError as e:
        print(f"Error reading file {file_path}: {e}")

    # Append the column of sigma_perp values to the matrix if not empty
    if sigma_perp_values:
        sigma_perp_matrix.append(sigma_perp_values)
    else:
        print(f"No valid Sigma_perp values found in file: {file_path}")

perp4 = np.array(sigma_perp_matrix).T


# Generate the list of file names based on the pattern
file_names1 = [f'1.{i}.5.xml.out' for i in range(240, 481, 5)]
file_names2 = [f'1.{i}.5.xml.out' for i in range(242, 478, 5)]
# Combine the lists
combined_file_names = file_names1 + file_names2


# Remove duplicates (optional)
combined_file_names = list(set(combined_file_names))

# Sort by the number after the first decimal point
sorted_file_names = sorted(combined_file_names, key=lambda x: int(x.split('.')[1]))

# Initialize an empty list to store sigma_perp columns
sigma_perp_matrix = []

# Loop through each file name
for combined_file_name in combined_file_names:
    sigma_perp_values = []

    # Construct the full path to the file
    file_path = os.path.join(directory, combined_file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    try:
        # Open the file and read its contents
        with open(file_path, 'r') as file:
            # Search for the header containing "Sigma_perp"
            for line in file:
                if 'Sigma_perp' in line:
                    # Identify the column index for 'Sigma_perp'
                    header_columns = line.split()
                    sigma_perp_index = header_columns.index('Sigma_perp')
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
                if len(values) > sigma_perp_index:
                    sigma_perp = values[sigma_perp_index]
                    try:
                        sigma_perp_values.append(float(sigma_perp))  # Convert to float
                    except ValueError as e:
                        print(f"Error converting '{sigma_perp}' to float in file: {file_path}. Error: {e}")

    except IOError as e:
        print(f"Error reading file {file_path}: {e}")

    # Append the column of sigma_perp values to the matrix if not empty
    if sigma_perp_values:
        sigma_perp_matrix.append(sigma_perp_values)
    else:
        print(f"No valid Sigma_perp values found in file: {file_path}")

perp5 = np.array(sigma_perp_matrix).T