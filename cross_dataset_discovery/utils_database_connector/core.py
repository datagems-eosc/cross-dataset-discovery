import os
from collections import defaultdict

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlglot import expressions, parse_one
from sqlglot.errors import ParseError


class Database:
    def __init__(
        self, database: str, max_execution_time: int = 180, specific_schema: str = None
    ):
        """
        Initialize the database connector. There are two types of databases supported: PostgreSQL, MySQL.
        The configuration of the database will be obtained from the utils_configs component.

        Args:
            database: the database name
            max_execution_time: the maximum execution time for a query in seconds. Default is 180s.
        """
        self.config = self._get_database_from_name(database)
        self.max_execution_time = max_execution_time
        self.specific_schema = specific_schema

        if "TEST" in os.environ:
            hostname = self.config.test_hostname
        elif "DEV" in os.environ:
            hostname = "localhost"
        else:  # pragma: no cover
            hostname = self.config.hostname

        self.connection_uri = (
            f"{self.config.type}+{self.config.driver}://{self.config.username}:"
            f"{self.config.password}@{hostname}:{self.config.port}/{self.config.name}"
        )

        if self.config.type == "postgresql":
            conn_args = {
                "options": f"-c statement_timeout={max_execution_time * 1000}"
                + (f" -c search_path={specific_schema}" if specific_schema else "")
            }
        elif self.config.type == "mysql":
            if specific_schema is not None:
                logger.warning(
                    "Query executor for MySQL does not support specific schema selection"
                )
            conn_args = {"read_timeout": max_execution_time}
        else:
            raise ValueError("Invalid database type")

        self.engine = create_engine(self.connection_uri, connect_args=conn_args)

        self.schemas = (
            ",".join(["'" + k + "'" for k in self.config.schemas])
            if specific_schema is None
            else f"'{specific_schema}'"
        )

    def _parse_query(self, query: str, limit: int, order_by_rand=False, only_read=True):
        pars = parse_one(query)
        if only_read and not isinstance(
            pars,
            (
                expressions.Select,
                expressions.Union,
                expressions.Intersect,
                expressions.Except,
            ),
        ):
            raise ValueError(
                "Database executor only accepts SELECT queries by default. "
                "If you want to execute other types of queries, set only_read=False."
            )

        if order_by_rand:
            if self.config.type != "mysql":
                pars = pars.order_by("random()")
            else:
                pars = pars.order_by("rand()")

        if limit not in (-1, 0):
            pars = pars.limit(limit)
        sql = pars.sql(dialect="mysql" if self.config.type == "mysql" else "postgres")
        # print(sql)
        return sql

    def execute(
        self,
        sql: str,
        limit: int = 500,
        order_by_rand: bool = False,
        fix_dates: bool = False,
        dates_format: str = "%d/%m/%Y",
        is_read: bool = True,
    ) -> pd.DataFrame | dict:
        """
        Execute a given SQL query.

        Note: By default the limit parameter is applied ignoring any limit set in the query from the user.
        To avoid applying a limit, set the limit parameter to -1.

        Args:
            sql: the sql query
            limit: the maximum number of rows to return. To ignore the limit parameter, set it to -1
            order_by_rand: whether to order the results randomly
            fix_dates: whether to fix the dates format
            dates_format: the dates format
            is_read: whether the query will read results from the database

        Returns:
            results: the results of the query or a dictionary with an error message
        """

        try:
            query = self._parse_query(sql, limit, order_by_rand, is_read)

            with self.engine.begin() as conn:
                df = pd.read_sql(text(query), con=conn)
            conn.close()
            self.engine.dispose()

            if fix_dates:
                mask = df.astype(str).apply(
                    lambda x: x.str.match(r"(\d{2,4}-\d{2}-\d{2,4})+").all()
                )
                df.loc[:, mask] = (  # type: ignore
                    df.loc[:, mask]  # type: ignore
                    .apply(pd.to_datetime)
                    .apply(lambda x: x.dt.strftime(dates_format))
                )
        except SQLAlchemyError as e:
            logger.error(f"sqlalchemy error {str(e.__dict__['orig'])}")
            return {"error": str(e.__dict__["orig"])}
        except ParseError as e:
            if len(e.errors) > 0:
                logger.error(f"parse error {e.errors[0]['description']}")
                return {"error": e.errors[0]["description"]}
            logger.error(f"parse error {str(e)}")
            return {"error": str(e)}
        except RuntimeError as e:
            logger.error(f"runtime error {str(e)}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"other exception: {e} {sql}")
            return {"error": f"Something went wrong with your query. Error: {e}"}
        return df

    def executemany(self, sql: str, data: list):
        """
        Execute many SQL queries in a batch (for bulk INSERT, UPDATE, or DELETE operations)

        Args:
            sql: the SQL query template
            data: list of dictionaries containing the data to be used in the query
        """
        try:
            with self.engine.begin() as conn:
                conn.execute(text(sql), data)
            return {"status": "success"}
        except SQLAlchemyError as e:
            return {"error": str(e.__dict__["orig"])}
        except Exception as e:
            print("other exception", e)
            return {"error": "Something went wrong with your query."}

        # TODO: Parse the query for the specific database type (PostgreSQL or MySQL)

    def get_tables_and_columns(self, blacklist_tables: list = []) -> dict:
        """
        Return the schema of the database

        Args:
            blacklist_tables: the tables to exclude from the results


        Examples:
            ```
            {
                'tables': ['table1', 'table2'],
                'columns': ['table1.column1', 'table1.column2', 'table2.column1'],
                'table': {
                    'table1': [0, 1],
                    'table2': [2]
                }
            }
            ```
        """
        q = f"""
            SELECT table_name,column_name
            FROM information_schema.COLUMNS
            WHERE table_schema in ({self.schemas})
        """  # nosec B608
        if (
            len(blacklist_tables) == 0 and len(self.config.blacklist_tables) > 0
        ):  # blacklist not provided, use default
            blacklist_tables = self.config.blacklist_tables

        if len(blacklist_tables) > 0:
            blacklist = " AND ".join(
                ["table_name not like '" + k + "'" for k in blacklist_tables]
            )
            q += " AND " + blacklist

        results = self.execute(q, limit=0)
        return self._parse_tables_and_columns(results)

    @staticmethod
    def _parse_tables_and_columns(results) -> dict:
        column_id = 0
        parsed = {"tables": [], "columns": [], "table": {}}

        for _, row in results.iterrows():
            table, column = row

            if table not in parsed["tables"]:
                parsed["tables"].append(table)
                parsed["table"][table] = []

            parsed["columns"].append(table + "." + column)
            parsed["table"][table].append(column_id)

            column_id += 1

        return parsed

    def get_types_of_db(self) -> dict:
        """
        Return the types of the columns of the database
        """
        ret_types = defaultdict(dict)
        if self.config.type == "postgresql":
            query = f"""
                SELECT table_name, column_name, data_type
                FROM information_schema.COLUMNS
                WHERE table_schema='{self.specific_schema if self.specific_schema else 'public'}'
                  AND table_name NOT ILIKE 'pg_%'
                  AND table_name NOT ILIKE 'sql_%';
            """  # nosec B608
            results = self.execute(query, limit=0)
            for _, row in results.iterrows():
                table, column, data_type = row
                data_type = data_type.replace("character varying", "varchar").upper()
                ret_types[table][column] = data_type
        else:
            raise ValueError(
                f"This method is only available for PostgreSQL databases, not {self.config.type}"
            )
        return ret_types

    def get_primary_keys(self) -> dict:
        """
        Return the primary keys of the database
        """
        ret_pks = defaultdict(list)
        if self.config.type == "postgresql":
            query = f"""
                    SELECT
                        tc.table_name,
                        kcu.column_name
                    FROM
                        information_schema.table_constraints AS tc
                        JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                    WHERE tc.constraint_type = 'PRIMARY KEY'
                        and tc.table_schema = '{self.specific_schema if self.specific_schema else 'public'}' ;
                """  # nosec B608

            results = self.execute(query, limit=0)
            for _, row in results.iterrows():
                table, column = row
                ret_pks[table].append(column)
        else:
            raise ValueError(
                f"This method is only available for PostgreSQL databases, not {self.config.type}"
            )

        return ret_pks

    def get_foreign_keys(self) -> dict:
        """
        Return the foreign keys of the database
        """
        ret_foreign_keys = {}
        if self.config.type == "postgresql":
            query = f"""
            SELECT
                tc.table_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema='{self.specific_schema if self.specific_schema else 'public'}'
            """  # nosec B608
            results = self.execute(query, limit=0)
            for _, row in results.iterrows():
                table, column, foreign_table, foreign_column = row
                if table in ret_foreign_keys and column in ret_foreign_keys[table]:
                    ret_foreign_keys[table][column].append(
                        {
                            "foreign_table": foreign_table,
                            "foreign_column": foreign_column,
                        }
                    )
                else:
                    ret_foreign_keys[table] = {
                        column: [
                            {
                                "foreign_table": foreign_table,
                                "foreign_column": foreign_column,
                            }
                        ]
                    }
        else:
            pass

        return ret_foreign_keys

    def get_joins(self) -> dict:
        """
        Return the joins for the database

        Examples:
            ```
            {
                'table1': {
                    'table2': 'table1.column1=table2.column1',
                    'table3': 'table1.column2=table3.column2'
                },
            }
            ```
        """
        if self.config.type != "mysql":
            query = f"""
            SELECT
                tc.table_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM
                information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY' and tc.table_schema in ({self.schemas})
            """  # nosec B608
        else:
            # Note: THis probably does not work. Check the query in get_foreign_keys
            query = f"""
            SELECT
                TABLE_NAME,
                COLUMN_NAME,
                REFERENCED_TABLE_NAME,
                REFERENCED_COLUMN_NAME
            FROM
                INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE REFERENCED_COLUMN_NAME is not null
            AND CONSTRAINT_SCHEMA in ({self.schemas})
            """  # nosec B608
        results = self.execute(query, limit=0)
        return self._parse_joins(results)

    @staticmethod
    def _parse_joins(results) -> dict:
        joins = {}
        for _, join in results.iterrows():
            for i in [0, 2]:
                thisTable = join[i]
                otherTable = join[0] if i == 2 else join[2]

                if thisTable not in joins:
                    joins[thisTable] = {}

                if otherTable not in joins[thisTable]:
                    joins[thisTable][otherTable] = []

                condition = join[0] + "." + join[1] + "=" + join[2] + "." + join[3]
                joins[thisTable][otherTable].append(condition)

        for tableA, valA in joins.items():
            for tableB, valB in valA.items():
                joins[tableA][tableB] = " AND ".join(valB)

        return joins

    def get_query_cost(self, query: str) -> float:
        """
        Gets the raw plan from EXPLAIN command of the input query and outputs the plan in a python dictionary

        Args:
            query (str): The SQL query

        Returns:
            Float: The execution estimated plan cost
        """

        explain_query = f"EXPLAIN (VERBOSE TRUE, SUMMARY TRUE, FORMAT json) {query}"
        plan = self.execute(sql=explain_query)

        # *** Explanation to extract the runtime estimation
        # 1. The EXPLAIN command has cardinality equal to 1, so the output is in the first row (first [0])
        # 2. The EXPLAIN returns a list encapsulating the whole plan, with one element, which is a JSON (second [0])
        # 3. Within the JSON there are two keys: a) the execution plan and b) the Planning Time:
        # 4. The structure of the execution plan is to nest the children of each node in a list, starting from the root.
        #   PostgresSQL computes the estimation in a bottom-up fashion, meaning that the first scan has starting
        #   estimate equal to 0.0 and the top node of the plan (root of the tree) has the cumulative estimates,
        #   thus for the whole plan.
        return float(plan[0][0]["Plan"]["Total Cost"])


if __name__ == "__main__":
    # q = """
    # SELECT
    #         a.fullname, a.orcid,
    #         COUNT(r.id) AS publication_num,
    #         STRING_AGG(r.id, '; ') AS publication_ids
    #     FROM author a
    #     LEFT JOIN result_author ra ON a.id = ra.author_id
    #     LEFT JOIN result r ON ra.result_id = r.id
    #     WHERE a.id = '00011ab1bc9af9fbfbabd4d8cca6fa76'
    #     GROUP BY a.id, a.fullname, a.orcid;
    # """
    # q = """
    # INSERT INTO table_name (column1, column2, column3)
    # VALUES (value1, value2, value3);
    # """
    q = """
    SELECT *
        FROM (
            SELECT r.title, r.publication_date
            FROM result r
            WHERE (r.keywords ILIKE '%recommender%' OR r.title ILIKE '%recommender%' OR r.description ILIKE '%recommender%') AND r.type = 'publication'
            ORDER BY r.publication_date DESC
         ) AS r
        UNION
        SELECT *
        FROM (
            SELECT r.title, r.publication_date
            FROM result r
            WHERE (r.keywords ILIKE '%recommender%' OR r.title ILIKE '%recommender%' OR r.description ILIKE '%recommender%') AND r.type = 'publication'
            ORDER BY r.publication_date
         ) AS r;
    """
    print(Database("fc4eosc").get_foreign_keys())
    # print(Database("fc4eosc", specific_schema="fc4eosc_subset").execute(q, limit=10))
    # db = Database("cordis", max_execution_time=0.0001)
    # print(db.execute("SELECT COUNT(*) FROM projects"))
