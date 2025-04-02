use dotenvy::dotenv;           // Allows loading environment variables from a .env file
use reqwest::Client;           // HTTP client for making requests (used for the LLM API calls)
use std::env;                  // Work with environment variables
use std::collections::HashMap; // For storing key-value mappings (e.g., ground truth data)
use std::path::Path;           // For handling file paths
use calamine::{open_workbook, Reader, Xlsx}; // For reading Excel (.xlsx) files
use xlsxwriter::*;             // For writing Excel (.xlsx) files
use serde::{Deserialize};      // For de/serializing JSON (used in LLM API responses)
use regex::Regex;              // For pattern matching and text post-processing
use chrono::Utc;               // For handling date and time
use std::fs::OpenOptions;      // For file operations
use std::io::{Write, BufWriter}; // For writing results (like markdown) in a buffered manner

// Struct to capture the response from Ollama LLM API.
#[derive(Debug, Deserialize)]
struct OllamaResponse {
    response: String,
}

// The main function, using Tokio's async runtime because we make async HTTP calls.
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load environment variables from .env file.
    dotenv().ok();

    // 2. Retrieve environment variables or provide defaults/fallbacks if not set.
    let rag_folders = env::var("RAG_FOLDER")
        .unwrap_or_else(|_| {
            eprintln!("⚠️ Warning: RAG_FOLDER not found in environment.");
            String::new()
        })
        // RAG_FOLDER can contain multiple folders separated by commas
        .split(',')
        .map(|s| s.trim().to_string())
        .collect::<Vec<String>>();

    let ground_truth_file = env::var("GROUND_TRUTH_FILE")
        .unwrap_or_else(|_| {
            eprintln!("⚠️ Warning: GROUND_TRUTH_FILE not found in environment. Using default.");
            "./data/references/ground_truth_v4.xlsx".to_string()
        });

    let output_xlsx = env::var("OUTPUT_XLSX")
        .unwrap_or_else(|_| {
            eprintln!("⚠️ Warning: OUTPUT_XLSX not found in environment. Using default.");
            "./data/results/final_results_modified.xlsx".to_string()
        });

    let ollama_url = env::var("OLLAMA_URL")
        .unwrap_or_else(|_| {
            eprintln!("⚠️ Warning: OLLAMA_URL not found in environment. Using default.");
            "http://localhost:11434/api/generate".to_string()
        });

    let ollama_model = env::var("OLLAMA_MODEL")
        .unwrap_or_else(|_| {
            eprintln!("⚠️ Warning: OLLAMA_MODEL not found in environment. Using default.");
            "llama3.2".to_string()
        });

    let eval_prompt = env::var("EVAL_PROMPT")
        .unwrap_or_else(|_| {
            eprintln!("⚠️ Warning: EVAL_PROMPT not found in environment. Using default.");
            "Is the generated answer correct based on the ground truth? Reply with 'Yes' or 'No' only.".to_string()
        });

    // 3. Load the ground truth data from an Excel file and store in a HashMap.
    let ground_truth_map = load_ground_truth(&ground_truth_file)?;

    // 4. Create an HTTP client for calling the LLM endpoints.
    let client = Client::new();

    // This vector will hold all rows from each RAG folder's results.
    let mut all_data = Vec::new();

    // 5. Iterate over each folder in RAG_FOLDER and read the results_230pages_groundtruth2.0_deepseek_generation.xlsx in that folder.
    for folder in rag_folders.iter() {
        let excel_path = format!("../{}/data/results.xlsx", folder);
        println!("Loading: {}", excel_path);

        // Check if the file exists before attempting to read it.
        if Path::new(&excel_path).exists() {
            println!("Processing: {}", excel_path);
            match read_excel_data(&excel_path, &ground_truth_map) {
                Ok(data) => {
                    if data.is_empty() {
                        println!("⚠️ Warning: No rows found in {}", excel_path);
                    } else {
                        println!("✅ Successfully read {} rows from {}", data.len(), excel_path);
                        all_data.extend(data);
                    }
                }
                Err(e) => eprintln!("❌ Error reading {}: {}", excel_path, e),
            }
        } else {
            println!("❌ File does not exist: {}", excel_path);
        }
    }

    // 6. If no data was found at all, there's nothing to process, so exit gracefully.
    if all_data.is_empty() {
        eprintln!("No valid Excel data found.");
        return Ok(());
    }

    // 7. Evaluate correctness of generated answers using the LLM (Ollama).

    // Modify this loop to use `call_ollama_llm_with_majority_vote`.
    let mut correctness = Vec::new();
    for row in all_data.iter() {
        // Create a prompt that compares ground truth vs generated answer.
        eprintln!("CURRENT EVAL");
        eprintln!("{}", row.ground_truth);
        eprintln!("{}", row.generated_answer);

        let correctness_prompt = format!(
            "Consider the question: {}\nGround truth: {}\nGenerated answer: {}\n\n{}",
            row.question, row.ground_truth, row.generated_answer, eval_prompt
        );
        eprintln!("{}", correctness_prompt);

        // Use majority voting for the LLM responses.
        match call_ollama_llm_with_majority_vote(&client, &correctness_prompt, &ollama_url, &ollama_model).await {
            Ok(response) => {
                // We then post-process the majority response to a standardized "true" / "false" / "unknown".
                let post_processed = postprocess_response(response.trim());
                correctness.push(post_processed);
            }
            Err(e) => {
                eprintln!("Validation error: {:?}", e);
                correctness.push("Error validating response".to_string());
            }
        }
        eprintln!("{:?}", correctness);
    }


    // 8. Write the final merged results into a single Excel file, including the correctness column.
    write_results_to_excel(&output_xlsx, &all_data, &correctness)?;

    // 9. Calculate accuracy grouped by model, based on the correctness data.
    let accuracy_results = calculate_model_accuracy(&all_data, &correctness, &ollama_model);

    // 10. Append these accuracy results to RESULTS.md for historical tracking.
    append_results_to_markdown(&accuracy_results)?;

    println!("✅ Merged results saved to {}", &output_xlsx);
    Ok(())
}

// A struct for holding all relevant data from each row in results_230pages_groundtruth2.0_deepseek_generation.xlsx.
#[derive(Debug)]
struct ExcelRow {
    branch_name: String,
    model_embedding: String,
    model_name: String,
    qid: String,
    question: String,
    ground_truth: String,
    generated_answer: String,
    retrieved_contexts: Vec<String>,
    elapsed_seconds: f64,
    date: String,
}

// Function to load ground truth data from an Excel file.
// Returns a HashMap where keys are questions and values are the ground truth answers.
fn load_ground_truth(file_path: &str) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    let mut workbook: Xlsx<_> = open_workbook(file_path)?;
    let mut ground_truth_map = HashMap::new();

    // Attempt to read the worksheet named "Sheet1".
    if let Some(Ok(range)) = workbook.worksheet_range("Sheet1") {
        eprintln!("Loading Ground Truth: Reading Sheet1...");

        // Skip the header row and iterate over each subsequent row.
        for row in range.rows().skip(1) {
            // We expect the columns: ID | Question | GroundTruth in the sheet.
            if let (Some(_id), Some(question), Some(ground_truth)) = (row.get(0), row.get(1), row.get(2)) {
                eprintln!("{}", ground_truth);
                ground_truth_map.insert(question.to_string(), ground_truth.to_string());
            }
        }
    } else {
        eprintln!("Error: Could not find Sheet1 in ground truth file.");
    }

    Ok(ground_truth_map)
}

// Function to read data from an Excel file named results_230pages_groundtruth2.0_deepseek_generation.xlsx in each folder,
// and match each question with its corresponding ground truth.
fn read_excel_data(file_path: &str, ground_truth_map: &HashMap<String, String>) 
    -> Result<Vec<ExcelRow>, Box<dyn std::error::Error>> 
{
    let mut workbook: Xlsx<_> = open_workbook(file_path)?;
    let mut data = Vec::new();

    // Try to read "Sheet1" from the workbook.
    if let Some(Ok(range)) = workbook.worksheet_range("Sheet1") {
        // Skip the header row and read the subsequent data rows.
        for row in range.rows().skip(1) {
            // Column indices:
            //   0: branch_name
            //   1: model_name
            //   2: qid
            //   3: question
            //   4: generated_answer
            //   5..14: retrieved contexts (10 columns)
            //   15: elapsed_seconds
            //   16: date
            if row.len() < 17 {
                // If there aren't enough columns, skip this row to avoid panic.
                continue;
            }

            // Extract the retrieved contexts as a vector of strings.
            let retrieved_contexts: Vec<String> = row.iter().skip(5).take(10)
                .map(|cell| cell.to_string())
                .collect();

            // **Handle the `elapsed_seconds` field properly**
            let elapsed_seconds = match row.get(16) {
                Some(calamine::DataType::String(value)) => {
                    value.trim().parse::<f64>().unwrap_or_else(|_| {
                        eprintln!("⚠️ Warning: Could not parse elapsed_seconds string: '{}'. Defaulting to 0.0.", value);
                        0.0
                    })
                }
                Some(calamine::DataType::Float(value)) => *value, // Extract float value directly
                Some(calamine::DataType::Int(value)) => *value as f64, // Convert integer to f64
                Some(calamine::DataType::Empty) => {
                    eprintln!("⚠️ Warning: elapsed_seconds is empty. Defaulting to 0.0.");
                    0.0
                }
                _ => {
                    eprintln!("⚠️ Warning: elapsed_seconds is of an unrecognized type. Defaulting to 0.0.");
                    0.0
                }
            };

            // Use pattern matching to safely get each required cell.
            if let (Some(branch_name),
                    Some(model_embedding),
                    Some(model_name),
                    Some(qid),
                    Some(question),
                    Some(generated_answer),
                    Some(date)
                ) = (
                    row.get(0),
                    row.get(1),
                    row.get(2),
                    row.get(3),
                    row.get(4),
                    row.get(5),
                    row.get(17)
                ) {

                // Convert the question to a string so we can look it up in the ground_truth_map.
                let question_str = question.to_string();
                // If the question isn't found, default to "N/A".
                let ground_truth = ground_truth_map
                    .get(&question_str)
                    .cloned()
                    .unwrap_or_else(|| "N/A".to_string());

                // Build the ExcelRow struct with all the needed info.
                data.push(ExcelRow {
                    branch_name: branch_name.to_string(),
                    model_embedding: model_embedding.to_string(),
                    model_name: model_name.to_string(),
                    qid: qid.to_string(),
                    question: question_str,
                    ground_truth,
                    generated_answer: generated_answer.to_string(),
                    retrieved_contexts,
                    elapsed_seconds, // ** Use the parsed value here **
                    date: date.to_string(),
                });
            }
        }
    } else {
        eprintln!("Error: Could not find Sheet1 in {}", file_path);
    }

    Ok(data)
}

// Writes the final merged results (across all RAG folders) into a single Excel file,
// including a column with the correctness classification.
fn write_results_to_excel(
    file_path: &str,
    data: &[ExcelRow],
    correctness: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    // Create a new workbook.
    let workbook = Workbook::new(file_path)?;
    // Create a new worksheet in the workbook.
    let mut sheet = workbook.add_worksheet(None)?;

    // Write the header row.
    sheet.write_string(0, 0, "Branch name", None)?;
    sheet.write_string(0, 1, "Model embedding", None)?;
    sheet.write_string(0, 2, "Model name", None)?;
    sheet.write_string(0, 3, "QID", None)?;
    sheet.write_string(0, 4, "Question", None)?;
    sheet.write_string(0, 5, "Ground Truth", None)?;
    sheet.write_string(0, 6, "Generated Answer", None)?;

    // Write headers for the retrieved contexts (columns 6..15).
    for i in 0..10 {
        let col_name = format!("Top {} retrieved", i + 1);
        sheet.write_string(0, 7 + i as u16, &col_name, None)?;
    }

    // Write headers for elapsed seconds, date, and correctness (columns 16..18).
    sheet.write_string(0, 16, "Elapsed seconds for experiment", None)?;
    sheet.write_string(0, 17, "Date", None)?;
    sheet.write_string(0, 18, "Correct", None)?;

    // Populate each row from the data slice.
    for (i, (row, correct)) in data.iter().zip(correctness.iter()).enumerate() {
        let row_num: u32 = (i + 1) as u32;

        sheet.write_string(row_num, 0, &row.branch_name, None)?;
        sheet.write_string(row_num, 1, &row.model_embedding, None)?;
        sheet.write_string(row_num, 2, &row.model_name, None)?;
        sheet.write_string(row_num, 3, &row.qid, None)?;
        sheet.write_string(row_num, 4, &row.question, None)?;
        sheet.write_string(row_num, 5, &row.ground_truth, None)?;
        sheet.write_string(row_num, 6, &row.generated_answer, None)?;

        // Fill in the retrieved contexts in their respective columns.
        for (j, retrieved) in row.retrieved_contexts.iter().enumerate() {
            sheet.write_string(row_num, (7 + j) as u16, retrieved, None)?;
        }

        // Write the elapsed seconds, date, and correctness classification.
        sheet.write_number(row_num, 16, row.elapsed_seconds, None)?;
        sheet.write_string(row_num, 17, &row.date, None)?;
        sheet.write_string(row_num, 18, correct, None)?;
    }

    // Close the workbook, finishing the write process.
    workbook.close()?;
    Ok(())
}

// Calls the Ollama LLM with a specified prompt, model, and temperature=0 (deterministic).
async fn call_ollama_llm(
    client: &Client,
    prompt: &str,
    url: &str,
    model: &str
) -> Result<String, reqwest::Error> {
    // JSON body with the prompt, specifying no streaming and temperature=0.
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "options": {
            "temperature": 0.0
          },
        "temperature": 0.0
    });

    // Perform the HTTP POST to the specified URL.
    let resp = client
        .post(url)
        .json(&body)
        .send()
        .await?
        .error_for_status()? // Propagate errors if the status is not success
        .json::<OllamaResponse>()
        .await?;

    // Return the .response field from the JSON response body.
    Ok(resp.response)
}

// Calls the Ollama LLM 5 times and returns the most common response.
async fn call_ollama_llm_with_majority_vote(
    client: &Client,
    prompt: &str,
    url: &str,
    model: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut responses = Vec::new();

    // Make 5 requests to the LLM.
    for _ in 0..9 {
        match call_ollama_llm(client, prompt, url, model).await {
            Ok(response) => responses.push(response.trim().to_string()),
            Err(e) => {
                eprintln!("Error during LLM call: {:?}", e);
                responses.push("Error".to_string()); // Add "Error" for failed requests.
            }
        }
    }

    // Count the frequency of each response.
    let mut frequency_map = HashMap::new();
    for response in responses {
        *frequency_map.entry(response).or_insert(0) += 1;
    }

    // Determine the most common response.
    let most_common_response = frequency_map
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(response, _)| response)
        .unwrap_or_else(|| "No valid response".to_string());

    Ok(most_common_response)
}

// Post-processes the LLM response so we can record "true", "false", or "unknown."
// This is used after the LLM's evaluation prompt to standardize the format.
fn postprocess_response(response: &str) -> String {
    // Convert the response to lowercase for uniform processing.
    let response_lower = response.to_lowercase();

    // Exact matches for "true" or "false".
    if response_lower == "true" {
        return "true".to_string();
    } else if response_lower == "false" {
        return "false".to_string();
    }

    // Otherwise, use regex to look for the words "true" or "false" in the text.
    let re_true = Regex::new(r"\btrue\b").unwrap();
    let re_false = Regex::new(r"\bfalse\b").unwrap();

    if re_true.is_match(&response_lower) {
        "true".to_string()
    } else if re_false.is_match(&response_lower) {
        "false".to_string()
    } else {
        // If neither is found, default to "unknown".
        "unknown".to_string()
    }
}

// Calculates model accuracy by grouping the results by model name,
// counting how many were marked "true" vs the total, and returns a
// HashMap from model name to percentage accuracy.
fn calculate_model_accuracy(
    data: &[ExcelRow],
    correctness: &[String],
    ollama_model: &String
) -> HashMap<String, (f64, String)> {
    let mut model_counts: HashMap<String, (i32, i32)> = HashMap::new();

    for (row, correct) in data.iter().zip(correctness.iter()) {
        let model_name = row.model_name.clone();
        let entry = model_counts.entry(model_name).or_insert((0, 0));
        entry.0 += 1;
        if correct == "true" {
            entry.1 += 1;
        }
    }

    let mut accuracy_results = HashMap::new();
    for (model, (total, correct)) in model_counts {
        let accuracy = if total > 0 {
            (correct as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        accuracy_results.insert(model, (accuracy, ollama_model.clone()));
    }

    accuracy_results
}

// Appends the calculated model accuracy to a markdown file named RESULTS.md
// to keep track of historical results.
fn append_results_to_markdown(accuracy_results: &HashMap<String, (f64, String)>) -> std::io::Result<()> {
    let file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("./RESULTS.md")?;

    let mut writer = BufWriter::new(file);

    for (model, (accuracy, eval_model)) in accuracy_results {
        let today_date = Utc::now().format("%Y-%m-%d").to_string();
        let line = format!("\n| {} | {} | {:.2} | {} |", model, today_date, accuracy, eval_model);
        writer.write_all(line.as_bytes())?;
    }

    writer.flush()?;
    Ok(())
}