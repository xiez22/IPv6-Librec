package com.ligongzzz.librec;

import android.content.Intent;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;

public class InfoActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_info);

        Intent intent = getIntent();
        ((TextView)findViewById(R.id.infoTitle)).setText(intent.getStringExtra("name"));
        ((TextView)findViewById(R.id.infoAuthor)).setText(intent.getStringExtra("author"));
        ((TextView)findViewById(R.id.infoCode)).setText("分类号："+intent.getStringExtra("code"));
        ((TextView)findViewById(R.id.infoYear)).setText("出版年："+intent.getStringExtra("year"));
        ((TextView)findViewById(R.id.infoBorrow)).setText("借阅次数："+intent.getStringExtra("borrow_cnt"));
        ((TextView)findViewById(R.id.infoPulisher)).setText("出版社："+intent.getStringExtra("publisher"));
    }
}
