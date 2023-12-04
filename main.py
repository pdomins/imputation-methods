import pandas as pd

from utils.noise_utils import add_noise, add_noise_to_column


def main():
    data = {'A': [1.0, 2.0, 3.0, 4.0],
            'B': [5.0, 6.0, 7.0, 8.0]}
    df = pd.DataFrame(data)
    print(df)
    print(add_noise(df))
    df['A'] = add_noise_to_column(df, column_name='A')
    df['B'] = add_noise_to_column(df, column_name='B')
    print(df)


if __name__ == '__main__':
    main()
