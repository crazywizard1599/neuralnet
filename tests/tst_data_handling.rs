use neuralnet::data_handling::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_csv_basic() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "name,age\nAlice,30\nBob,25").unwrap();

        let records = read_csv(file.path()).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0], vec!["Alice".to_string(), "30".to_string()]);
        assert_eq!(records[1], vec!["Bob".to_string(), "25".to_string()]);
    }

    #[test]
    fn test_read_csv_empty_file() {
        let file = NamedTempFile::new().unwrap();
        let records = read_csv(file.path()).unwrap();
        assert_eq!(records.len(), 0);
    }

    #[test]
    fn test_read_csv_invalid_path() {
        let result = read_csv("non_existent_file.csv");
        assert!(result.is_err());
    }

    #[test]
    fn test_read_json_basic() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"name": "Alice", "age": 30}}"#).unwrap();

        let json = read_json(file.path()).unwrap();
        assert_eq!(json["name"], "Alice");
        assert_eq!(json["age"], 30);
    }

    #[test]
    fn test_read_json_invalid_json() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"name": "Alice", "age": }}"#).unwrap();

        let result = read_json(file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_read_json_invalid_path() {
        let result = read_json("non_existent_file.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_read_excel_basic() {
        // Create a temporary xlsx file with one sheet and some data
        let mut file = NamedTempFile::new().unwrap();
        // Minimal Excel file in binary format is complex to generate manually,
        // so this test checks that the function returns an error for a non-Excel file.
        writeln!(file, "name,age\nAlice,30\nBob,25").unwrap();

        let result = read_excel(file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_read_excel_invalid_path() {
        let result = read_excel("non_existent_file.xlsx");
        assert!(result.is_err());
    }

    #[test]
    fn test_read_excel_empty_file() {
        let file = NamedTempFile::new().unwrap();
        let result = read_excel(file.path());
        assert!(result.is_err());
    }
}

