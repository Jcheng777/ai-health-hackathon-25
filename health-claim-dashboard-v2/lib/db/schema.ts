import { pgTable, serial, integer, varchar, numeric, text, jsonb, timestamp } from 'drizzle-orm/pg-core';

export const claim = pgTable('claim', {
  id: serial('id').primaryKey(),
  providerId: integer('provider_id'),
  procedureCode: varchar('procedure_code', { length: 255 }),
  diagnosisCode: varchar('diagnosis_code', { length: 255 }),
  billedAmount: numeric('billed_amount'),
  insuranceType: varchar('insurance_type', { length: 255 }),
  additionalInfo: text('additional_info'),

  // Prediction fields
  prediction: varchar('prediction_new', { length: 255 }),
  confidenceScore: numeric('confidence_score'),
  likelihoodPercent: numeric('likelihood_percent'),
  denialReasons: jsonb('denial_reasons'),
  acceptedReasons: jsonb('accepted_reasons'),
  nextSteps: jsonb('next_steps'),
  analysisDetails: jsonb('analysis_details'),
  dateCreated: text('date_created'),
}); 