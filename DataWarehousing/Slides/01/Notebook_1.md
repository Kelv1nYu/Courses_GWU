# 名词解释

1. Database - A collection of related data

2. Data - Known facts that can be recorded and have an implicit meaning

3. Mini-world - Some part of the real world about which data is stored in a database

4. Database Management System(DBMS) - A software package/system to facilitate the creation and maintenance of a computerized database

5. Database System - The DBMS software together with the data itself (Sometimes applications are also included)

6. Queries - Access different parts of data and formulate the result of a request

7. Transactions - Read some data and "update" certain values or generate new data and store that in the database

8. Meta-data -  Define and describe the data and relationships between tables in the database

9. Program-data Independence - Insulation between programs and data

10. Data Model - A set of concepts to describe the structure of a database, the operations for manipulating these structures, and certain constraints that the database should obey

11. Concurrency Control Strategies - Features of a database that allow several users access to the same data item at the same time

12. Data type - Determine the sort of data permitted in a field

13. Data uniqueness - Ensure that no duplicates are entered

14. Database constraint - A restriction that determines what is allowed to be entered or edited in a table

15. Schema - Logical structure of data

# 概念

1. A database represents some aspect fo the real world, sometimes called mini-world or the universe of discourse (UoD).

2. A database is a logically coherent collection of data with some inherent meaning.

3. A database is designed, built, and populated with data for a specific purpose. It has an intended group of users and some preconceived applications in which these users are interested.

4. Application interact with a database by generating.

5. Applications must now allow unauthorized users to access data.

6. Applications must keep up with changing user requirements against the database.

7. Concurrency control within the DBMS guarantees that each transaction is correctly executed and aborted.

8. Recovery subsystem ensures each completed transaction has its effect permanently recorded in the database.

9. OLTP (Online Transaction Processing) is a major part of database applications which allows hundreds of concurrent transactions to execute per second.

10. Constructs are used to define the database structure, typically include elements (and their data types) and groups of elements (entity, record, table) and relationships among such groups.

11. Constraints are restrictions on valid data and must be enforced at all times.

# 图解数据库环境

1. Users or programmers access data through Application Programs or Queries.

2. DBMS Software can receive the process queries or programs sent by applications (users) , then handle those messages and access stored data.

3. There are two types of data will be stored in database, meta-data and data.

4. Meta-data is the definition or information of data.

**Some newer system such as a few NoSQL systems need no meta-data -- they store the data definition within its structure making it self describing**

# DBMS 功能

1. Define a particular database in terms of its data types, structures, and constraints;

2. Construct or Load the initial database contents on a secondary storage medium;

3. Manipulating the database;

4. Processing and Sharing by a set of concurrent users and application programs;

5. Keeping all data valid and consistent;

6. Provide protection or security measures to prevent unauthorized access;

7. Provide "Active" processing to take internal actions on data;

8. Presentation and Visualization of data;

9. Maintenance of the database and associated programs over the lifetime of the database application.

# 使用 Database 的好处

1. Data independence and efficient access

2. Reduced application development time

3. Data integrity and security

4. Data administration

5. Concurrent access and crash recovery

# 使用 Database Approach 的好处

1. Controlling redundancy in data storage and in development and maintenance efforts

2. Restricing unauthorized access to data -- only the DBA (Database administrators) stuff uses privileged commands and facilities

3. Providing Storage Structures for efficient Query Processing

4. Providing optimization of queries

5. Backup and Recovery

6. Representing Complex Relationships

7. Efficient integrity constraints on the database

8. Potential for enforcing standards

9. Reduced application development time

10. Flexibility to change data structures

11. Availability of current information

12. Economies of scale

# 不必使用 DBMS 的情况

Costs

Unnecessary

Infeasible

Suffice

# Database Schemas 和 Instances

Database Schema is the description of a database, includes descriptions of a database structure, data types, and the constraints on the database and will changes very infrequently.

Database Instance is the actual data stored in the database at a particular moment in time, include the collection of all the data in the database.

# DBMS 语言

Data Definition Laguage (DDL) - Used by the DBA and database designers to specify the conceptual schema of a database

Data Manipulation Language (DML) - Used to specify database retrievals and updates

Data Control Language (DCL) - Similar to computer programming language used to control access to the database

# 关系型数据库

## 定义

1. Relation consists of 2 parts: Schema and Instance.

2. Schema specifies the name of the relation and the name of each column.

3. Instance represents a table with rows and columns.

4. Each row (tuple) represents a record of related data values.

5. Each row (tuple) corresponds to a real-world entity or relationship.

6. Each row (tuple) defined by the set of attributes.

7. Each column (attribute) holds a corresponding value for each row.

## 特征

1. Every cell in the data table contains only one value;

2. All of the values in a single column must be of the same type;

3. Each column has a unique name;

4. Order of either columns or rows is not significant.

## 约束

**Constraints determine which values are permissible and which are not in the database**

### Key constraints

**no two tuples can have the same combination of values for their attributes**

#### Primary Key (主键) Constraints

If a relation has several candidate keys, one is chosen (underline) arbitrarily to be the primary key.

主键要保证唯一性，即一张表只能有一个主键，且主键一列的值必须互不相同且不为空。

若可通过某一列属性的某一个值来获得一整行的数据，则可将这列属性称为主键。 (例，我们可以通过学生id来从学生信息表中确定该学生的信息，因为学生id是唯一且互不相同的)

#### Foreign Key (外键)

外键用于维护引用完整性。

被引用的表的主键将被作为引用表的外键。

例，学生表中若存在对课程表的引用，课程表的主键（课程id）则会被作为学生表的外键，由学生表中的课程id指向课程表的课程id。

### Entity integrity constraints

1. Every table must have primary key.

2. Neither Primary Key or any of its part can be NULL.

### Referential integrity constraints

Used to specify a relationship among tuples in two relations -- referencing relation and the referenced relation.

### Domain 

Defining valid set of values for attributes.

## 更新操作

INSERT / DELETE / MODIFY (UPDATE)

不同的操作可能组合在一起，并且会引起其他的操作。

## 更新操作产生完整性违规

1. Cancel the operation that causes the violation (RESTRICT or REJECT option)

2. Perform the operation but inform the user of the violation

3. Trigger additional updates so the violation is corrected (CASCADE option, SET NULL option)

4. Execute a user-specified error-correction routine 

**任何操作都可能引起完整性违规，所以要特别注意完整性的约束条件。**