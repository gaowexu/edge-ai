import { expect as expectCDK, matchTemplate, MatchStyle } from '@aws-cdk/assert';
import * as cdk from '@aws-cdk/core';
import * as EdgeAiWithOta from '../lib/edge-ai-with-ota-stack';

test('Empty Stack', () => {
    const app = new cdk.App();
    // WHEN
    const stack = new EdgeAiWithOta.EdgeAiWithOtaStack(app, 'MyTestStack');
    // THEN
    expectCDK(stack).to(matchTemplate({
      "Resources": {}
    }, MatchStyle.EXACT))
});
