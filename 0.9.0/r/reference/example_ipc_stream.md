# Example Arrow IPC Data

An example stream that can be used for testing or examples.

## Usage

``` r
example_ipc_stream(compression = c("none", "zstd", "lz4"))
```

## Arguments

- compression:

  One of "none", "zstd", or "lz4"

## Value

A raw vector that can be passed to
[`read_nanoarrow()`](read_nanoarrow.md)

## Examples

``` r
as.data.frame(read_nanoarrow(example_ipc_stream()))
#>   some_col
#> 1        0
#> 2        1
#> 3        2
```
