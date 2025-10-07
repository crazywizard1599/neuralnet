use std::error::Error;
use std::fs::File;
use std::path::Path;
use csv::ReaderBuilder;
use serde_json::Value;
use calamine::{open_workbook_auto, Reader, DataType};

/// Reads a CSV file from the given path and returns its records as a vector of string vectors.
/// 
/// # Arguments
/// * `path` - Path to the CSV file.
/// 
/// # Returns
/// * `Ok(Vec<Vec<String>>)` - Each inner vector represents a row of the CSV file.
/// * `Err(Box<dyn Error>)` - If the file cannot be read or parsed.
/// 
pub fn read_csv<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<String>>, Box<dyn Error>> {
    // Open the file at the given path
    let file = File::open(path)?;
    // Create a CSV reader with headers enabled
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    let mut records = Vec::new();

    // Iterate over each record (row) in the CSV
    for result in rdr.records() {
        let record = result?;
        // Convert each field to String and collect into a vector
        records.push(record.iter().map(|s| s.to_string()).collect());
    }

    Ok(records)
}

/// Reads a JSON file from the given path and returns its contents as a serde_json::Value.
/// 
/// # Arguments
/// * `path` - Path to the JSON file.
/// 
/// # Returns
/// * `Ok(Value)` - Parsed JSON data.
/// * `Err(Box<dyn Error>)` - If the file cannot be read or parsed.
///
pub fn read_json<P: AsRef<Path>>(path: P) -> Result<Value, Box<dyn Error>> {
    // Open the file at the given path
    let file = File::open(path)?;
    // Parse the file contents as JSON
    let json: Value = serde_json::from_reader(file)?;
    Ok(json)
}

/// Reads an Excel file from the given path and returns its first sheet as a vector of string vectors.
/// 
/// # Arguments
/// * `path` - Path to the Excel file (.xls, .xlsx, etc.).
/// 
/// # Returns
/// * `Ok(Vec<Vec<String>>)` - Each inner vector represents a row of the first sheet.
/// * `Err(Box<dyn Error>)` - If the file cannot be read or parsed.
/// 
pub fn read_excel<P: AsRef<Path>>(path: P) -> Result<Vec<Vec<String>>, Box<dyn Error>> {
    // Open the Excel workbook at the given path
    let mut workbook = open_workbook_auto(path)?;
    // Get the names of all sheets in the workbook
    let sheet_names = workbook.sheet_names().to_owned();
    if sheet_names.is_empty() {
        return Err("No sheets found in Excel file".into());
    }
    // Try to get the range (data) of the first sheet
    let range = workbook.worksheet_range(&sheet_names[0])
        .ok_or("Cannot find the first sheet")??;

    let mut records = Vec::new();
    // Iterate over each row in the sheet
    for row in range.rows() {
        // Convert each cell to String based on its type
        let record = row.iter()
            .map(|cell| match cell {
                DataType::Empty => "".to_string(),
                DataType::String(s) => s.clone(),
                DataType::Float(f) => f.to_string(),
                DataType::Int(i) => i.to_string(),
                DataType::Bool(b) => b.to_string(),
                DataType::Error(e) => format!("Error: {:?}", e),
                DataType::DateTime(f) => f.to_string(),
            })
            .collect();
        records.push(record);
    }
    Ok(records)
}

