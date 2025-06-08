import { NextRequest, NextResponse } from 'next/server';
import { OpenAI } from 'openai';
import * as pdfParse from 'pdf-parse';
import { createWorker } from 'tesseract.js';
import sharp from 'sharp';

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Supported file types
const SUPPORTED_FILE_TYPES = {
  'application/pdf': 'pdf',
  'image/png': 'png',
  'image/jpeg': 'jpeg',
  'image/jpg': 'jpg'
};

async function extractTextFromImage(buffer: Buffer): Promise<string> {
  const worker = await createWorker();
  await worker.loadLanguage('eng');
  await worker.initialize('eng');
  
  // Preprocess image for better OCR
  const processedImage = await sharp(buffer)
    .grayscale()
    .normalize()
    .toBuffer();
  
  const { data: { text } } = await worker.recognize(processedImage);
  await worker.terminate();
  return text;
}

async function extractTextFromFile(file: File): Promise<string> {
  const arrayBuffer = await file.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);
  const fileType = SUPPORTED_FILE_TYPES[file.type as keyof typeof SUPPORTED_FILE_TYPES];

  if (!fileType) {
    throw new Error('Unsupported file type');
  }

  if (fileType === 'pdf') {
    const pdfData = await pdfParse(buffer);
    return pdfData.text;
  } else {
    // Handle image files (png, jpeg, jpg)
    return await extractTextFromImage(buffer);
  }
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;

    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      );
    }

    // Check if file type is supported
    if (!SUPPORTED_FILE_TYPES[file.type as keyof typeof SUPPORTED_FILE_TYPES]) {
      return NextResponse.json(
        { error: 'Unsupported file type. Please upload a PDF, PNG, JPEG, or JPG file.' },
        { status: 400 }
      );
    }

    // Extract text from file
    const extractedText = await extractTextFromFile(file);

    // Process with GPT
    const completion = await openai.chat.completions.create({
      model: "gpt-4-turbo-preview",
      messages: [
        {
          role: "system",
          content: `You are a medical billing assistant. Extract key structured information from a CMS-1500 health insurance claim form. 

Please return your output in valid JSON format with these fields:

{
  "patient_name": "",
  "patient_dob": "",
  "patient_sex": "",
  "insured_id": "",
  "insured_name": "",
  "insured_dob": "",
  "insured_sex": "",
  "relationship_to_insured": "",
  "insurance_plan_name": "",
  "prior_authorization_number": "",
  "referring_provider_name": "",
  "referring_provider_npi": "",
  "rendering_provider_name": "",
  "rendering_provider_npi": "",
  "date_of_service": "",
  "diagnosis_codes": [],
  "procedure_codes": [
    {
      "cpt": "",
      "modifier": "",
      "diagnosis_pointer": [],
      "charges": "",
      "units": "",
      "rendering_provider_npi": ""
    }
  ],
  "billed_amount_total": "",
  "facility_name": "",
  "facility_address": "",
  "tax_id": "",
  "claim_signature_date": ""
}`
        },
        {
          role: "user",
          content: `Here is the OCR or parsed text of the claim form:\n\n${extractedText}`
        }
      ],
      response_format: { type: "json_object" }
    });

    const structuredData = JSON.parse(completion.choices[0].message.content);

    return NextResponse.json({
      success: true,
      data: structuredData
    });

  } catch (error) {
    console.error('Error processing file:', error);
    return NextResponse.json(
      { error: 'Error processing file' },
      { status: 500 }
    );
  }
} 