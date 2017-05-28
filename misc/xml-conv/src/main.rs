extern crate xml;
extern crate glob;
extern crate regex;

use std::fs::File;
use std::io::BufReader;
use std::collections::HashMap;
use glob::glob;
use regex::Regex;
use xml::reader::{EventReader, XmlEvent};

fn main() {
    for file in glob("icsi_mr_transcr/transcripts/*.mrt").expect("Failed") {
        let file = File::open(file.unwrap()).unwrap();
        let file = BufReader::new(file);
        let parser = EventReader::new(file);
        let mut in_segment = false;
        let mut participant: String = String::from("");
        let mut map = HashMap::new();
        let r = Regex::new("[-.,?!\"@]").unwrap();
        for e in parser {
            match e {
                Ok(XmlEvent::StartElement { name, attributes, ..}) => {
                    if name.local_name == "Segment" {
                        if let Some(attr) = attributes.iter().find(|r| r.name.local_name == "Participant") {
                            in_segment = true;
                            participant = attr.value.to_owned()
                        }
                    }
                },
                Ok(XmlEvent::EndElement { name, ..}) => {
                    if name.local_name == "Segment" {
                        in_segment = false;
                    }
                },
                Ok(XmlEvent::Characters(ref string)) if in_segment => {
                    let entry = map.entry(participant.to_string()).or_insert_with(|| String::new());
                    (*entry).push_str(" ");
                    (*entry).push_str(&r.replace_all(&string.trim().to_lowercase(), ""));
                }
                _ => {}
            }
        }
        for (k,v) in map {
            println!("{}", v);
        }
    }
}