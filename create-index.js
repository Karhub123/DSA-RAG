import * as dotenv from 'dotenv';
dotenv.config();
import { Pinecone } from '@pinecone-database/pinecone';

const pc = new Pinecone();

await pc.createIndex({
  name: 'dsa-rag',
  dimension: 3072,
  metric: 'cosine',
  spec: {
    serverless: { cloud: 'aws', region: 'us-east-1' },
  },
  waitUntilReady: true,
});

console.log('index created');