import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import (shapiro, levene, ttest_ind, mannwhitneyu,
                          kruskal, f_oneway, chi2_contingency,
                          spearmanr, pearsonr)
from itertools import combinations
from statsmodels.tsa.seasonal import seasonal_decompose


class SalesDataBuilder:

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.df = None

    def _load_raw_tables(self):
        olap = f"{self.base_path}/OLAP"
        self.customer = pd.read_csv(f"{olap}/DimCustomer.csv")
        self.geography = pd.read_csv(f"{olap}/DimGeography.csv")
        self.orderPriority = pd.read_csv(f"{olap}/DimOrderPriority.csv")
        self.product = pd.read_csv(f"{olap}/DimProduct.csv")
        self.shipMode = pd.read_csv(f"{olap}/DimShipMode.csv")
        self.factSales = pd.read_csv(f"{olap}/FactSales.csv")
        self.date = pd.read_excel(f"{self.base_path}/DimDate.xlsx")

    def _print_raw_quality_report(self):
        tables = [("date", self.date), ("product", self.product), ("customer", self.customer),
                  ("geography", self.geography), ("shipMode", self.shipMode),
                  ("orderPriority", self.orderPriority), ("factSales", self.factSales)]
        for name, tbl in tables:
            print(f"--- {name} ---")
            print(tbl.shape)
            print(tbl.isnull().sum().sum(), "null values")
            print()

    def _print_key_dtype_report(self):
        print("FactSales dtypes:\n",
              self.factSales[['Product ID', 'CustomerID', 'GeoKey', 'ShipModeID', 'PriorityKey']].dtypes)
        print("\nProduct ProductID dtype:", self.product['ProductID'].dtype)
        print("Customer CustomerID dtype:", self.customer['CustomerID'].dtype)
        print("Geography GeoKey dtype:", self.geography['GeoKey'].dtype)
        print("ShipMode ShipModeID dtype:", self.shipMode['ShipModeID'].dtype)
        print("OrderPriority PriorityKey dtype:", self.orderPriority['PriorityKey'].dtype)

    def _merge_simple_dims(self, verbose=True):
        df = self.factSales.copy()
        n0 = len(df)
        if verbose:
            print("Initial number of rows:", n0)

        df = df.merge(self.product, left_on='Product ID', right_on='ProductID', how='left')
        if verbose:
            print("After Product:", len(df))

        df = df.merge(self.customer, on='CustomerID', how='left')
        if verbose:
            print("After Customer:", len(df))

        df = df.merge(self.geography, on='GeoKey', how='left')
        if verbose:
            print("After Geography:", len(df))

        df = df.merge(self.shipMode, on='ShipModeID', how='left')
        if verbose:
            print("After ShipMode:", len(df))

        df = df.merge(self.orderPriority, on='PriorityKey', how='left')
        if verbose:
            print("After OrderPriority:", len(df))

        self._n0 = n0
        return df

    def _merge_date_dim(self, df, verbose=True):
        df['OrderDate'] = pd.to_datetime(df['OrderDate'])
        df['ShipDate'] = pd.to_datetime(df['ShipDate'])
        self.date['GregorianDate'] = pd.to_datetime(self.date['GregorianDate'])

        useful_date_cols = [
            'GregorianDate', 'GregorianYearInt', 'GregorianMonthNo', 'GregorianMonthName',
            'GregorianDayOfWeekName', 'SeasonName', 'IsGregorianLeap'
        ]
        date_subset = self.date[useful_date_cols]

        date_order = date_subset.add_prefix('Order_')
        df = df.merge(date_order, left_on='OrderDate', right_on='Order_GregorianDate', how='left')
        if verbose:
            print("After Order Date:", len(df))

        date_ship = date_subset.add_prefix('Ship_')
        df = df.merge(date_ship, left_on='ShipDate', right_on='Ship_GregorianDate', how='left')
        if verbose:
            print("After Ship Date:", len(df))

        assert len(df) == self._n0
        return df

    def _print_post_merge_report(self, df):
        print(df.shape)
        nulls = df.isnull().sum()
        print(nulls[nulls > 0])

    def build(self, verbose: bool = True) -> pd.DataFrame:
     
        self._load_raw_tables()
        if verbose:
            self._print_raw_quality_report()
            self._print_key_dtype_report()

        df = self._merge_simple_dims(verbose)
        df = self._merge_date_dim(df, verbose)

        if verbose:
            self._print_post_merge_report(df)

        df = df.drop(columns=['ProductID', 'Order_GregorianDate', 'Ship_GregorianDate'])

        if verbose:
            df.info()
            print(df.columns)
            print(df.dtypes)

        self.df = df
        return self.df



class SalesDataCleaner:


    CATEGORICAL_COLS = ['Segment', 'Category', 'Sub-Category', 'Region',
                        'Market', 'ShipMode', 'OrderPriority']

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def cast_dates(self):
        self.df['OrderDate'] = pd.to_datetime(self.df['OrderDate'])
        self.df['ShipDate'] = pd.to_datetime(self.df['ShipDate'])
        return self

    def check_duplicates(self):
        print("Duplicate rows:", self.df.duplicated().sum())
        print("SalesKey:", self.df['SalesKey'].duplicated().sum())
        return self

    def check_invalid_values(self):
        invalid_dates = self.df[self.df['ShipDate'] < self.df['OrderDate']]
        print("Rows with invalid dates:", len(invalid_dates))
        print("Invalid Quantity:", (self.df['Quantity'] < 0).sum())
        print("Invalid Sales:", (self.df['Sales'] < 0).sum())
        print("Invalid Discount:", ((self.df['Discount'] < 0) | (self.df['Discount'] > 1)).sum())
        return self

    def check_nulls(self):
        null_counts = self.df.isnull().sum()
        print(null_counts[null_counts > 0])
        return self

    def print_categorical_value_counts(self):
        for col in self.CATEGORICAL_COLS:
            print(f"--- {col} ---")
            print(self.df[col].value_counts())
            print()
        return self

    def standardize_segment(self):
        self.df['Segment'] = self.df['Segment'].str.strip().str.title()
        return self

    def strip_column_names(self):
        self.df.columns = self.df.columns.str.strip()
        return self

    def translate_season_names(self):
        season_map = {"بهار": "Spring", "تابستان": "Summer", "پاییز": "Fall", "زمستان": "Winter"}
        self.df['Order_SeasonName'] = self.df['Order_SeasonName'].map(season_map)
        self.df['Ship_SeasonName'] = self.df['Ship_SeasonName'].map(season_map)
        return self

    def fix_is_returned_bug(self):
        self.df['IsReturned'] = self.df['IsReturned'].apply(lambda x: 'Yes' if x != 'No' else 'No')
        return self

    def clean(self, verbose: bool = True) -> pd.DataFrame:
        """Runs every cleaning step in the exact order of cells [15]-[24]."""
        self.cast_dates()
        if verbose:
            self.check_duplicates()
            self.check_invalid_values()
            self.check_nulls()
            self.print_categorical_value_counts()
        self.standardize_segment()
        self.strip_column_names()
        if verbose:
            print(list(self.df.columns))
        self.translate_season_names()
        self.fix_is_returned_bug()
        return self.df

    def save_to_csv(self, path: str):
        self.df.to_csv(path, index=False)
        print("shape :", self.df.shape)



class EDAVisualizer:
  
    def __init__(self, df: pd.DataFrame):
        self.df = df
        sns.set_style('whitegrid')

    def show_head(self):
        return self.df.head()

    def describe_numeric(self):
        return self.df.describe().T

    def describe_categorical(self):
        return self.df.describe(include='object').T

    def plot_numeric_distributions(self, numeric_cols):
        for col in numeric_cols:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            sns.histplot(self.df[col], kde=True, ax=axes[0])
            axes[0].set_title(f'Histogram + KDE: {col}')
            axes[0].set_xlim(self.df[col].quantile(0.01), self.df[col].quantile(0.99))
            sns.boxplot(x=self.df[col], ax=axes[1])
            axes[1].set_title(f'Boxplot: {col}')
            stats.probplot(self.df[col], dist="norm", plot=axes[2])
            axes[2].set_title(f'Q-Q Plot: {col}')
            plt.tight_layout()
            plt.show()
            print(f"{col}: skewness={self.df[col].skew():.2f}, kurtosis={self.df[col].kurt():.2f}")

    def plot_categorical_frequencies(self, categorical_cols):
        for col in categorical_cols:
            fig, ax = plt.subplots(figsize=(10, 4))
            order = self.df[col].value_counts().index
            sns.countplot(data=self.df, y=col, order=order, ax=ax)
            ax.set_title(f'Frequency of {col}')
            plt.tight_layout()
            plt.show()
            print(self.df[col].value_counts(normalize=True) * 100)

    def plot_numeric_pairwise(self, numeric_cols):
        sns.pairplot(self.df[numeric_cols].sample(2000), diag_kind='kde')
        plt.show()

        corr_matrix = self.df[numeric_cols].corr(method='pearson')
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Pearson Correlation Matrix')
        plt.show()

    def plot_numeric_by_categories(self, cat_cols, value_col='Profit'):
        for cat_col in cat_cols:
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.boxplot(data=self.df, x=cat_col, y=value_col, ax=ax, showfliers=False)
            plt.xticks(rotation=45)
            ax.set_title(f'{value_col} by {cat_col} (outliers hidden for clarity)')
            plt.tight_layout()
            plt.show()

    def plot_categorical_crosstab_heatmap(self, col1, col2, cmap='YlGnBu'):
        cross_tab = pd.crosstab(self.df[col1], self.df[col2])
        print(cross_tab)
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap=cmap)
        plt.title(f'{col1} vs {col2}')
        plt.show()

    def plot_time_series_overview(self, date_col='OrderDate'):
        monthly = self.df.set_index(date_col).resample('MS')[['Sales', 'Profit']].sum()
        fig, ax = plt.subplots(figsize=(14, 5))
        monthly.plot(ax=ax)
        ax.set_title('Monthly Sales & Profit Trend')
        plt.show()

        seasonal = self.df.groupby('Order_SeasonName')['Sales'].sum().sort_values(ascending=False)
        sns.barplot(x=seasonal.index, y=seasonal.values)
        plt.title('Sales by Season')
        plt.show()

        weekday = self.df.groupby('Order_GregorianDayOfWeekName')['Sales'].mean()
        sns.barplot(x=weekday.index, y=weekday.values)
        plt.title('Average Sales by Day of Week')
        plt.xticks(rotation=45)
        plt.show()

        result = seasonal_decompose(monthly['Sales'], model='additive', period=12)
        result.plot()
        plt.show()

    def plot_top_countries_and_market_summary(self):
        top_countries = self.df.groupby('Country')['Sales'].sum().sort_values(ascending=False).head(15)
        sns.barplot(x=top_countries.values, y=top_countries.index)
        plt.title('Top 15 Countries by Sales')
        plt.show()

        market_summary = self.df.groupby('Market').agg(
            total_sales=('Sales', 'sum'),
            avg_profit=('Profit', 'mean'),
            order_count=('Order ID', 'nunique')
        ).sort_values('total_sales', ascending=False)
        print(market_summary)
        return market_summary

    @staticmethod
    def _iqr_outliers(series):
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return (series < lower) | (series > upper)

    def analyze_outliers(self, numeric_cols):
        for col in numeric_cols:
            mask = self._iqr_outliers(self.df[col])
            print(f"{col}: {mask.sum()} outliers ({mask.sum()/len(self.df)*100:.1f}%)")

        top_outliers = self.df[self._iqr_outliers(self.df['Sales'])].sort_values('Sales', ascending=False)
        print(top_outliers[['Order ID', 'ProductName', 'Sales', 'Quantity', 'Category']].head(20))

    def plot_violin_profit_by_segment(self):
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.violinplot(data=self.df, x='Segment', y='Profit', ax=ax, cut=0)
        ax.set_ylim(-200, 300)
        ax.set_title('Violin Plot: Profit Distribution by Segment')
        plt.tight_layout()
        plt.show()

    def plot_discount_by_category(self):
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.boxplot(data=self.df, x='Category', y='Discount', ax=ax, showfliers=False)
        ax.set_title('Discount Distribution by Category')
        plt.tight_layout()
        plt.show()

    def plot_lmplot_discount_profit_by_category(self):
        g = sns.lmplot(data=self.df.sample(5000, random_state=42), x='Discount', y='Profit',
                        col='Category', height=4, aspect=1, scatter_kws={'alpha': 0.2, 's': 10},
                        line_kws={'color': 'red'})
        g.set(ylim=(-500, 500))
        g.fig.suptitle('Discount vs Profit by Category (with regression line)', y=1.05)
        plt.show()

    def plot_shipmode_priority_heatmap(self):
        self.plot_categorical_crosstab_heatmap('ShipMode', 'OrderPriority', cmap='Blues')

    def plot_quarterly_trend_by_category(self):
        self.df['OrderDate'] = pd.to_datetime(self.df['OrderDate'])
        monthly_cat = self.df.set_index('OrderDate').groupby('Category').resample('QS')['Sales'].sum().reset_index()

        fig, ax = plt.subplots(figsize=(13, 5))
        sns.lineplot(data=monthly_cat, x='OrderDate', y='Sales', hue='Category', marker='o', ax=ax)
        ax.set_title('Quarterly Sales Trend by Category')
        plt.tight_layout()
        plt.show()

    def plot_pareto_customers(self):
        cust_sales = self.df.groupby('CustomerName')['Sales'].sum().sort_values(ascending=False).reset_index()
        cust_sales['cum_pct'] = cust_sales['Sales'].cumsum() / cust_sales['Sales'].sum() * 100
        cust_sales['cust_pct'] = (np.arange(1, len(cust_sales) + 1) / len(cust_sales)) * 100

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(cust_sales['cust_pct'], cust_sales['cum_pct'], color='#4C72B0')
        ax1.axhline(80, color='red', linestyle='--', alpha=0.7, label='80% of Sales')
        ax1.set_xlabel('% of Customers (sorted by sales)')
        ax1.set_ylabel('Cumulative % of Total Sales')
        ax1.set_title('Pareto Analysis: Do 20% of Customers Drive 80% of Sales?')
        ax1.legend()
        plt.tight_layout()
        plt.show()

        pct_customers_for_80 = cust_sales[cust_sales['cum_pct'] <= 80]['cust_pct'].max()
        print(f"The percentage of customers who account for 80% of sales: {pct_customers_for_80:.1f}%")
        return pct_customers_for_80

    def plot_pearson_vs_spearman(self, numeric_cols):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        sns.heatmap(self.df[numeric_cols].corr(method='pearson'), annot=True, cmap='coolwarm',
                    center=0, ax=axes[0], fmt='.2f')
        axes[0].set_title('Pearson Correlation')
        sns.heatmap(self.df[numeric_cols].corr(method='spearman'), annot=True, cmap='coolwarm',
                    center=0, ax=axes[1], fmt='.2f')
        axes[1].set_title('Spearman Correlation')
        plt.tight_layout()
        plt.show()


    def plot_correlation_clustermap(self, numeric_cols, method='spearman'):
        corr = self.df[numeric_cols].corr(method=method)
        g = sns.clustermap(corr, annot=True, cmap='coolwarm', center=0, figsize=(8, 8), fmt='.2f')
        g.fig.suptitle(f'{method.title()} Correlation — Clustered', y=1.02)
        plt.show()

    def plot_small_multiples_trend_by_region(self, value_col='Sales', date_col='OrderDate'):

        import matplotlib.dates as mdates
 
        df = self.df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        grouped = df.set_index(date_col).groupby('Region').resample('QS')[value_col].sum().reset_index()
 
        g = sns.relplot(data=grouped, x=date_col, y=value_col, col='Region', col_wrap=4,
                         kind='line', height=2.5, facet_kws={'sharey': False})
        g.fig.suptitle(f'Quarterly {value_col} Trend — Small Multiples by Region', y=1.03)
 
        for ax in g.axes.flatten():
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.tick_params(axis='x', rotation=45)
 
        plt.show()

    def plot_ridge_profit_by_category(self, clip=(-500, 500)):
        cats = self.df['Category'].unique()
        colors = sns.color_palette('viridis', len(cats))
        fig, ax = plt.subplots(figsize=(10, 5))
        for color, cat in zip(colors, cats):
            sub = self.df[self.df['Category'] == cat]['Profit']
            sns.kdeplot(sub, ax=ax, fill=True, alpha=0.5, color=color,
                        bw_adjust=0.5, clip=clip, label=cat)
        ax.set_title('Profit Density by Category (overlaid KDE)')
        ax.set_xlabel('Profit')
        ax.legend()
        plt.tight_layout()
        plt.show()



class HypothesisTester:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run_hypothesis_discount_vs_quantity(self):
        self.df['has_discount'] = self.df['Discount'].apply(lambda x: 'Discounted' if x > 0 else 'No discount')
        print(self.df['has_discount'].value_counts())
        print()

        group_d = self.df[self.df['has_discount'] == 'Discounted']['Quantity']
        group_nd = self.df[self.df['has_discount'] == 'No discount']['Quantity']
        print(self.df.groupby('has_discount')['Quantity'].describe())
        print()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.histplot(data=self.df, x='Quantity', hue='has_discount', element='step',
                     stat='density', common_norm=False, ax=axes[0], discrete=True)
        axes[0].set_title('Distribution of Quantity by Discount Group')
        sns.boxplot(data=self.df, x='has_discount', y='Quantity', ax=axes[1])
        axes[1].set_title('Quantity by Discount Group')
        plt.tight_layout()
        plt.show()

        for name, g in [('Discounted', group_d), ('No discount', group_nd)]:
            p = shapiro(g.sample(min(1000, len(g)), random_state=42))[1]
            normal = "It's normal" if p >= 0.05 else "not normal"
            print(f"Shapiro-Wilk [{name}]: p={p:.6f} → {normal}")

        stat_l, p_l = levene(group_d, group_nd)
        equal_var = p_l >= 0.05
        print(f"\nLevene's Test: statistic={stat_l:.4f}, p={p_l:.6f} → "
              f"Variances are {'equal' if equal_var else 'not equal'}")

        t_stat, p_ttest = ttest_ind(group_d, group_nd, equal_var=equal_var)
        print(f"\nIndependent t-test: t={t_stat:.4f}, p={p_ttest:.6f}")

        u_stat, p_mwu = mannwhitneyu(group_d, group_nd, alternative='two-sided')
        print(f"Mann-Whitney U: U={u_stat:.2f}, p={p_mwu:.6f}  ← Recommended test")

        mean_diff = group_d.mean() - group_nd.mean()
        pooled_std = np.sqrt((group_d.std() ** 2 + group_nd.std() ** 2) / 2)
        cohens_d = mean_diff / pooled_std
        print(f"\nMean Difference: {mean_diff:.4f}")
        print(f"Cohen's d: {cohens_d:.4f}  "
              f"({'negligible' if abs(cohens_d) < 0.2 else 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'})")

        return {"p_ttest": p_ttest, "p_mwu": p_mwu, "cohens_d": cohens_d}


    def run_hypothesis_profit_by_category(self):
        cats = self.df['Category'].unique()
        groups_cat = [self.df[self.df['Category'] == c]['Profit'].values for c in cats]

        print(self.df.groupby('Category')['Profit'].describe())
        print()

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.boxplot(data=self.df, x='Category', y='Profit', ax=ax, showfliers=False)
        ax.set_title('Profit by Category (without outliers for better clarity)')
        plt.tight_layout()
        plt.show()

        for cat in cats:
            sub = self.df[self.df['Category'] == cat]['Profit']
            sample = sub.sample(min(1000, len(sub)), random_state=42)
            p = shapiro(sample)[1]
            print(f"Shapiro [{cat}]: p={p:.6f} → {'Normal' if p >= 0.05 else 'Not normal'}")

        stat_l2, p_l2 = levene(*groups_cat)
        print(f"\nLevene: statistic={stat_l2:.4f}, p={p_l2:.6f}")

        stat_kw, p_kw = kruskal(*groups_cat)
        print(f"\nKruskal-Wallis: stat={stat_kw:.4f}, p={p_kw:.6f}")

        f_stat, p_anova = f_oneway(*groups_cat)
        print(f"One-way ANOVA (for comparison): F={f_stat:.4f}, p={p_anova:.6f}")

        print("\nPost-hoc pairwise (Mann-Whitney U + Bonferroni correction):")
        pairs = list(combinations(cats, 2))
        bonferroni_alpha = 0.05 / len(pairs)
        for c1, c2 in pairs:
            g1 = self.df[self.df['Category'] == c1]['Profit']
            g2 = self.df[self.df['Category'] == c2]['Profit']
            u, p = mannwhitneyu(g1, g2, alternative='two-sided')
            sig = "Significant" if p < bonferroni_alpha else "Not significant"
            print(f"  {c1} vs {c2}: p={p:.6f} (threshold={bonferroni_alpha:.5f}) → {sig} "
                  f"| mean1={g1.mean():.2f}, mean2={g2.mean():.2f}")

        return {"p_kruskal": p_kw, "p_anova": p_anova}


    def run_hypothesis_return_by_category(self):
        contingency = pd.crosstab(self.df['Category'], self.df['IsReturned'])
        print(contingency)
        print()

        return_rate = (contingency['Yes'] / (contingency['Yes'] + contingency['No']) * 100)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=return_rate.index, y=return_rate.values, ax=ax,
                    hue=return_rate.index, legend=False, palette='Reds_r')
        ax.set_ylabel('Return Rate (%)')
        ax.set_title('Return Rate by Category')
        plt.tight_layout()
        plt.show()

        print("Return Rate by Category (%):")
        print(return_rate.round(2))

        chi2, p_chi, dof, expected = chi2_contingency(contingency)
        print(f"\nMinimum expected count in contingency table: {expected.min():.1f}  "
              f"({'Acceptable (>=5)' if expected.min() >= 5 else 'Warning: Less than 5'})")

        print(f"\nChi-Square: X^2={chi2:.4f}, p-value={p_chi:.6f}, dof={dof}")

        n = contingency.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
        print(f"Cramér's V: {cramers_v:.4f}  "
              f"({'negligible' if cramers_v < 0.1 else 'small' if cramers_v < 0.3 else 'medium' if cramers_v < 0.5 else 'large'})")

        return {"chi2": chi2, "p": p_chi, "cramers_v": cramers_v}


    def run_hypothesis_discount_vs_profit(self):
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.scatterplot(data=self.df.sample(3000, random_state=42), x='Discount', y='Profit', alpha=0.35, ax=ax)
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.set_title('Discount vs Profit')
        plt.tight_layout()
        plt.show()

        for col in ['Discount', 'Profit']:
            p = shapiro(self.df[col].sample(1000, random_state=42))[1]
            print(f"Shapiro [{col}]: p={p:.6f} → {'Normal' if p >= 0.05 else 'Not normal'}")

        r, p_pearson = pearsonr(self.df['Discount'], self.df['Profit'])
        print(f"\nPearson r={r:.4f}, p-value={p_pearson:.6f}")

        rho, p_spearman = spearmanr(self.df['Discount'], self.df['Profit'])
        print(f"Spearman rho={rho:.4f}, p-value={p_spearman:.6f}  ← Recommended test (more reliable due to non-normality)")

        return {"pearson_r": r, "spearman_rho": rho}

    def check_normality(self, series, label="", sample_size=1000):
        sample = series.sample(min(sample_size, len(series)), random_state=42)
        stat, p = shapiro(sample)
        is_normal = p >= 0.05
        print(f"Shapiro-Wilk [{label}]: p={p:.6f} -> {'Normal' if is_normal else 'Not normal'}")
        return is_normal

    def check_variance_homogeneity(self, *groups):
        stat, p = levene(*groups)
        equal_var = p >= 0.05
        print(f"Levene's Test: statistic={stat:.4f}, p={p:.6f} -> "
              f"{'Equal variances' if equal_var else 'Unequal variances'}")
        return equal_var

    def compare_two_groups(self, group_col, value_col):
        levels = self.df[group_col].unique()
        assert len(levels) == 2, "This method requires exactly two distinct groups."
        g1 = self.df[self.df[group_col] == levels[0]][value_col]
        g2 = self.df[self.df[group_col] == levels[1]][value_col]
        print(self.df.groupby(group_col)[value_col].describe())
        self.check_normality(g1, levels[0])
        self.check_normality(g2, levels[1])
        equal_var = self.check_variance_homogeneity(g1, g2)
        t_stat, p_ttest = ttest_ind(g1, g2, equal_var=equal_var)
        u_stat, p_mwu = mannwhitneyu(g1, g2, alternative='two-sided')
        mean_diff = g1.mean() - g2.mean()
        pooled_std = np.sqrt((g1.std() ** 2 + g2.std() ** 2) / 2)
        cohens_d = mean_diff / pooled_std
        print(f"\nt-test: t={t_stat:.4f}, p={p_ttest:.6f}")
        print(f"Mann-Whitney U: U={u_stat:.2f}, p={p_mwu:.6f}")
        print(f"Cohen's d: {cohens_d:.4f}")
        return {"p_ttest": p_ttest, "p_mwu": p_mwu, "cohens_d": cohens_d}

    def compare_multiple_groups(self, group_col, value_col):
        levels = self.df[group_col].unique()
        groups = [self.df[self.df[group_col] == lvl][value_col].values for lvl in levels]
        for lvl, g in zip(levels, groups):
            self.check_normality(pd.Series(g), lvl)
        self.check_variance_homogeneity(*groups)
        stat_kw, p_kw = kruskal(*groups)
        f_stat, p_anova = f_oneway(*groups)
        print(f"\nKruskal-Wallis: stat={stat_kw:.4f}, p={p_kw:.6f}")
        print(f"One-way ANOVA: F={f_stat:.4f}, p={p_anova:.6f}")
        print("\nPost-hoc pairwise (Mann-Whitney U + Bonferroni):")
        pairs = list(combinations(levels, 2))
        alpha = 0.05 / len(pairs)
        for l1, l2 in pairs:
            g1 = self.df[self.df[group_col] == l1][value_col]
            g2 = self.df[self.df[group_col] == l2][value_col]
            u, p = mannwhitneyu(g1, g2, alternative='two-sided')
            sig = "Significant" if p < alpha else "Not significant"
            print(f"  {l1} vs {l2}: p={p:.6f} -> {sig}")
        return {"p_kruskal": p_kw, "p_anova": p_anova}

    def chi_square_test(self, col1, col2):
        contingency = pd.crosstab(self.df[col1], self.df[col2])
        chi2, p, dof, expected = chi2_contingency(contingency)
        n = contingency.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
        print(contingency)
        print(f"\nChi-Square: chi2={chi2:.4f}, p={p:.6f}, dof={dof}")
        print(f"Cramér's V: {cramers_v:.4f}")
        return {"chi2": chi2, "p": p, "cramers_v": cramers_v}

    def correlation_test(self, col1, col2):
        self.check_normality(self.df[col1], col1)
        self.check_normality(self.df[col2], col2)
        r, p_pearson = pearsonr(self.df[col1], self.df[col2])
        rho, p_spearman = spearmanr(self.df[col1], self.df[col2])
        print(f"\nPearson r={r:.4f}, p={p_pearson:.6f}")
        print(f"Spearman rho={rho:.4f}, p={p_spearman:.6f}")
        return {"pearson_r": r, "spearman_rho": rho}



class InteractiveEDA:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def scatter_sales_vs_profit(self, sample_size=5000):
        import plotly.express as px
        sample = self.df.sample(min(sample_size, len(self.df)), random_state=42)
        fig = px.scatter(
            sample, x='Sales', y='Profit', color='Category',
            hover_data=['ProductName', 'CustomerName', 'Order ID', 'Discount'],
            opacity=0.6, title='Sales vs Profit (interactive — hover for details, click legend to filter)'
        )
        fig.add_hline(y=0, line_dash='dash', line_color='red')
        fig.update_layout(height=550)
        fig.show()

    def treemap_category_subcategory(self):
        import plotly.express as px
        agg = self.df.groupby(['Category', 'Sub-Category']).agg(
            total_sales=('Sales', 'sum'),
            avg_profit=('Profit', 'mean')
        ).reset_index()
        fig = px.treemap(
            agg, path=['Category', 'Sub-Category'], values='total_sales',
            color='avg_profit', color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
            title='Sales by Category / Sub-Category (size=Sales, color=Avg Profit) — click to zoom'
        )
        fig.update_layout(height=600)
        fig.show()

    def sunburst_geography(self):
        import plotly.express as px
        agg = self.df.groupby(['Market', 'Region', 'Country'])['Sales'].sum().reset_index()
        fig = px.sunburst(
            agg, path=['Market', 'Region', 'Country'], values='Sales',
            title='Sales Breakdown: Market → Region → Country (click to drill down)'
        )
        fig.update_layout(height=650)
        fig.show()

    def monthly_trend_with_rangeslider(self, value_col='Sales'):
        import plotly.express as px
        df = self.df.copy()
        df['OrderDate'] = pd.to_datetime(df['OrderDate'])
        monthly = df.set_index('OrderDate').groupby('Category').resample('MS')[value_col].sum().reset_index()
        fig = px.line(
            monthly, x='OrderDate', y=value_col, color='Category',
            title=f'Monthly {value_col} by Category (drag the range slider to zoom)'
        )
        fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(height=550)
        fig.show()

    def choropleth_sales_by_country(self):
        import plotly.express as px
        agg = self.df.groupby('Country')['Sales'].sum().reset_index()
        fig = px.choropleth(
            agg, locations='Country', locationmode='country names', color='Sales',
            color_continuous_scale='Blues', title='Total Sales by Country (hover for values)'
        )
        fig.update_layout(height=550)
        fig.show()

    def parallel_categories_returns(self):
        import plotly.express as px
        fig = px.parallel_categories(
            self.df, dimensions=['Segment', 'Category', 'ShipMode', 'IsReturned'],
            title='Segment → Category → ShipMode → Returned (drag a band to highlight a flow)'
        )
        fig.update_layout(height=550)
        fig.show()