package com.ligongzzz.librec.ui.notifications;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.text.Html;
import android.text.method.LinkMovementMethod;
import android.text.util.Linkify;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.Observer;
import androidx.lifecycle.ViewModelProviders;
import com.ligongzzz.librec.R;

public class NotificationsFragment extends Fragment {

    private NotificationsViewModel notificationsViewModel;

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {
        notificationsViewModel =
                ViewModelProviders.of(this).get(NotificationsViewModel.class);
        final View root = inflater.inflate(R.layout.fragment_notifications, container, false);

        notificationsViewModel.getText().observe(getViewLifecycleOwner(), new Observer<String>() {
            @Override
            public void onChanged(@Nullable String s) {
                SharedPreferences sharedPreferences = getActivity().getSharedPreferences("data", Context.MODE_PRIVATE);
                String UserID = sharedPreferences.getString("UserID", "");

                if (!UserID.isEmpty()) {
                    ((EditText)root.findViewById(R.id.loginUserID)).setText(UserID);
                }

                String IP = sharedPreferences.getString("IP", "[2001:da8:270:2020:f816:3eff:fef7:817c]");

                String url = "<a href=\"http://" + IP + "/signup\">没有账号？注册一个！</a>";
                TextView textView = (TextView)root.findViewById(R.id.signupTextView);
                textView.setText(Html.fromHtml(url));
                textView.setMovementMethod(LinkMovementMethod.getInstance());

                ((TextView)root.findViewById(R.id.IPShowTextView)).setText("当前IP："+IP);
                ((EditText)root.findViewById(R.id.IP)).setText(IP);
            }
        });

        return root;
    }
}